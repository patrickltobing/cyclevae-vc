#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
# based on PyTorch implementation for WaveNet vocoder by Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import argparse
import logging
import math
import os
import sys
from distutils.util import strtobool

import numpy as np
import torch
import torch.multiprocessing as mp

from utils import find_files
from utils import read_hdf5
from utils import read_txt
from utils import check_hdf5
from utils import write_hdf5
from feature_extract_vc import low_cut_filter, convert_f0, mod_pow, read_wav, analyze_range
from feature_extract_vc import convert_continuos_f0, low_pass_filter
from feature_extract_vc import spc2npow, extfrm

from scipy.io import wavfile

from gru_vae import GRU_RNN, sampling_vae_batch
from dtw_c import dtw_c as dtw

import pysptk as ps
import pyworld as pw
from pysptk.synthesis import MLSADF

#FS = 16000
FS = 22050
N_GPUS = 1
SHIFT_MS = 5.0
#MCEP_ALPHA = 0.41000000000000003
MCEP_ALPHA = 0.455
FFTL = 1024
IRLEN = 1024
INTERVALS = 10
SEED = 1
GPU_DEVICE = 0
VERBOSE = 1
LP_CUTOFF = 20



def main():
    parser = argparse.ArgumentParser()
    # decode setting
    parser.add_argument("--waveforms", required=True,
                        type=str, help="list or directory of source eval feat files")
    parser.add_argument("--waveforms_trg", required=True,
                        type=str, help="list or directory of source eval feat files")
    parser.add_argument("--feats", required=True,
                        type=str, help="list or directory of source eval feat files")
    parser.add_argument("--feats_trg", default=None,
                        type=str, help="list or directory of target eval feat files")
    parser.add_argument("--stats_src", required=True,
                        type=str, help="hdf5 file including source statistics")
    parser.add_argument("--stats_trg", required=True,
                        type=str, help="hdf5 file including target statistics")
    parser.add_argument("--stats_jnt",
                        type=str, help="hdf5 file including target statistics")
    parser.add_argument("--model", required=True,
                        type=str, help="GRU_RNN model file")
    parser.add_argument("--config", required=True,
                        type=str, help="GRU_RNN configure file")
    parser.add_argument("--outdir", required=True,
                        type=str, help="directory to save generated samples")
    parser.add_argument("--fs", default=FS,
                        type=int, help="sampling rate")
    parser.add_argument("--n_gpus", default=N_GPUS,
                        type=int, help="number of gpus")
    parser.add_argument("--n_smpl_dec", default=300,
                        type=int, help="number of gpus")
    # other setting
    parser.add_argument("--shiftms", default=SHIFT_MS,
                        type=float, help="frame shift")
    parser.add_argument("--mcep_alpha", default=MCEP_ALPHA,
                        type=float, help="mcep alpha coeff.")
    parser.add_argument("--fftl", default=FFTL,
                        type=int, help="FFT length")
    parser.add_argument("--intervals", default=INTERVALS,
                        type=int, help="log interval")
    parser.add_argument("--seed", default=SEED,
                        type=int, help="seed number")
    parser.add_argument("--minf0", default=40,
                        type=int, help="seed number")
    parser.add_argument("--maxf0", default=700,
                        type=int, help="seed number")
    parser.add_argument("--minf0_trg", default=40,
                        type=int, help="seed number")
    parser.add_argument("--maxf0_trg", default=700,
                        type=int, help="seed number")
    parser.add_argument("--pow", default=-25.0,
                        type=float, help="seed number")
    parser.add_argument("--pow_trg", default=-25.0,
                        type=float, help="seed number")
    parser.add_argument("--GPU_device", default=GPU_DEVICE,
                        type=int, help="selection of GPU device")
    parser.add_argument("--verbose", default=VERBOSE,
                        type=int, help="log level")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]		= "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]	= str(args.GPU_device)

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # set log level
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    # fix seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load config
    config = torch.load(args.config)

    # get file list
    if os.path.isdir(args.feats):
        feat_list = sorted(find_files(args.feats, "*.h5"))
    elif os.path.isfile(args.feats):
        feat_list = read_txt(args.feats)
    else:
        logging.error("--feats should be directory or list.")
        sys.exit(1)
    if os.path.isdir(args.waveforms):
        wav_list = sorted(find_files(args.waveforms, "*.wav"))
    elif os.path.isfile(args.waveforms):
        wav_list = read_txt(args.waveforms)
    else:
        logging.error("--waveforms should be directory or list.")
        sys.exit(1)
    if os.path.isdir(args.waveforms_trg):
        wav_trg_list = sorted(find_files(args.waveforms_trg, "*.wav"))
    elif os.path.isfile(args.waveforms_trg):
        wav_trg_list = read_txt(args.waveforms_trg)
    else:
        logging.error("--waveforms_trg should be directory or list.")
        sys.exit(1)

    spk_src = os.path.basename(os.path.dirname(wav_list[0]))
    spk_trg = os.path.basename(os.path.dirname(wav_trg_list[0]))

    # define f0 statistics source
    f0_range_mean_src = read_hdf5(args.stats_src, "/lf0_range_mean")
    f0_range_std_src = read_hdf5(args.stats_src, "/lf0_range_std")
    logging.info(f0_range_mean_src)
    logging.info(f0_range_std_src)

    # define f0 statistics target
    f0_range_mean_trg = read_hdf5(args.stats_trg, "/lf0_range_mean")
    f0_range_std_trg = read_hdf5(args.stats_trg, "/lf0_range_std")
    logging.info(f0_range_mean_trg)
    logging.info(f0_range_std_trg)
    gv_mean_src = read_hdf5(args.stats_src, "/gv_range_mean")[1:]
    gv_mean_trg = read_hdf5(args.stats_trg, "/gv_range_mean")[1:]

    # prepare the file list for parallel decoding
    feat_lists = np.array_split(feat_list, args.n_gpus)
    feat_lists = [f_list.tolist() for f_list in feat_lists]
    wav_lists = np.array_split(wav_list, args.n_gpus)
    wav_lists = [f_list.tolist() for f_list in wav_lists]
    wav_trg_lists = np.array_split(wav_trg_list, args.n_gpus)
    wav_trg_lists = [f_list.tolist() for f_list in wav_trg_lists]

    ### GRU-RNN decoding ###
    def decode_RNN(feat_list, wav_list, wav_trg_list, gpu, cvlist=None, cvlist_src=None, cvgvlist=None, cvgvlist_src=None, difflist=None, diffgvlist=None, mcd_cvlist=None, mcdstd_cvlist=None, mcdpow_cvlist=None, mcdpowstd_cvlist=None, mcd_cvgvlist=None, mcdstd_cvgvlist=None, mcd_cvlist_src=None, mcdstd_cvlist_src=None, mcdpow_cvlist_src=None, mcdpowstd_cvlist_src=None, mcd_cvgvlist_src=None, mcdstd_cvgvlist_src=None, mcd_cvlist_trg=None, mcdstd_cvlist_trg=None, mcdpow_cvlist_trg=None, mcdpowstd_cvlist_trg=None, mcd_cvgvlist_trg=None, mcdstd_cvgvlist_trg=None, mcd_difflist=None, mcdstd_difflist=None, mcd_diffgvlist=None, mcdstd_diffgvlist=None, lat_dist_rmse_enc_list=None, lat_dist_cosim_enc_list=None, lat_dist_rmse_pri_list=None, lat_dist_cosim_pri_list=None):
        with torch.cuda.device(gpu):
            tmp_str = args.model.split('/')[1].split('_')
            mdl_name = tmp_str[2]+"_"+tmp_str[3]
            logging.info(mdl_name)
            string_path = mdl_name+"-"+str(config.n_cyc)+"-"+str(config.lat_dim)+"-"+str(torch.load(args.model)["iterations"])+"-"+spk_trg+"-"+str(args.n_smpl_dec)
            logging.info(string_path)
            string_mean = "/cvgv_mean_"+string_path
            cvgv_mean = read_hdf5(args.stats_src, string_mean)
            string_mean = "/cvgvsrc_mean_"+string_path
            cvgvsrc_mean = read_hdf5(args.stats_src, string_mean)
            string_mean = "/cvgvtrg_mean_"+string_path
            cvgvtrg_mean = read_hdf5(args.stats_src, string_mean)
            mean_trg = torch.FloatTensor(read_hdf5(args.stats_jnt, "/mean_feat_org_lf0_jnt")[config.stdim:]).cuda()
            std_trg = torch.FloatTensor(read_hdf5(args.stats_jnt, "/scale_feat_org_lf0_jnt")[config.stdim:]).cuda()
            # define model and load parameters
            logging.info(config)
            logging.info("model")
            with torch.no_grad():
                model_encoder = GRU_RNN(
                    in_dim=config.in_dim,
                    out_dim=config.lat_dim*2,
                    hidden_layers=config.hidden_layers,
                    hidden_units=config.hidden_units,
                    kernel_size=config.kernel_size,
                    dilation_size=config.dilation_size,
                    scale_out_flag=False)
                model_decoder = GRU_RNN(
                    in_dim=config.lat_dim+2,
                    out_dim=config.out_dim,
                    hidden_layers=config.hidden_layers,
                    hidden_units=config.hidden_units,
                    kernel_size=config.kernel_size,
                    dilation_size=config.dilation_size,
                    scale_in_flag=False)
                model_encoder.load_state_dict(torch.load(args.model)["model_encoder"])
                model_decoder.load_state_dict(torch.load(args.model)["model_decoder"])
                model_encoder.cuda()
                model_decoder.cuda()
                model_encoder.eval()
                model_decoder.eval()
                for param in model_encoder.parameters():
                    param.requires_grad = False
                for param in model_decoder.parameters():
                    param.requires_grad = False
                logging.info(model_encoder)
                logging.info(model_decoder)
                init_pp = np.zeros((1,1,config.lat_dim*2))
                y_in_pp = torch.FloatTensor(init_pp).cuda()
                y_in_src = y_in_trg = torch.unsqueeze(torch.unsqueeze((0-mean_trg)/std_trg,0),0)
            fs = args.fs
            fft_size = args.fftl
            mcep_dim = model_decoder.out_dim-1
            for feat_file, wav_file, wav_trg_file in zip(feat_list, wav_list, wav_trg_list):
                # convert mcep
                logging.info("cvmcep " + feat_file + " " + wav_file + " " + wav_trg_file)
                fs, x = read_wav(wav_file, cutoff=70)

                time_axis, f0, sp, ap = analyze_range(x, fs=fs, minf0=args.minf0, maxf0=args.maxf0, fperiod=args.shiftms, fftl=args.fftl)
                logging.info(sp.shape)

                mcep = ps.sp2mc(sp, mcep_dim, args.mcep_alpha)
                logging.info(mcep.shape)
                codeap = pw.code_aperiodicity(ap, fs)
                logging.info(codeap.shape)

                npow_src = spc2npow(sp)
                logging.info(npow_src.shape)
                _, spcidx_src = extfrm(mcep, npow_src, power_threshold=args.pow)
                spcidx_src = spcidx_src[0]
                logging.info(spcidx_src.shape)

                _, x_trg = read_wav(wav_trg_file, cutoff=70)
                _, f0_trg, sp_trg, ap_trg = analyze_range(x_trg, fs=fs, minf0=args.minf0_trg, maxf0=args.maxf0_trg, fperiod=args.shiftms, fftl=args.fftl)
                mcep_trg = ps.sp2mc(sp_trg, mcep_dim, args.mcep_alpha)
                logging.info(mcep_trg.shape)
                codeap_trg = pw.code_aperiodicity(ap_trg, fs)
                logging.info(codeap_trg.shape)
                npow_trg = spc2npow(sp_trg)
                logging.info(npow_trg.shape)
                mcepspc_trg, spcidx_trg = extfrm(mcep_trg, npow_trg, power_threshold=args.pow_trg)
                logging.info(mcepspc_trg.shape)
                spcidx_trg = spcidx_trg[0]
                logging.info(spcidx_trg.shape)

                uv, contf0 = convert_continuos_f0(np.array(f0))
                uv = np.expand_dims(uv, axis=-1)
                logging.info(uv.shape)
                cont_f0_lpf = low_pass_filter(contf0, int(1.0 / (5.0 * 0.001)), cutoff=20)
                logcontf0 = np.expand_dims(np.log(cont_f0_lpf), axis=-1)
                logging.info(logcontf0.shape)
                feat = np.c_[uv,logcontf0,codeap,mcep]
                logging.info(feat.shape)

                uv_trg, contf0_trg = convert_continuos_f0(np.array(f0_trg))
                uv_trg = np.expand_dims(uv_trg, axis=-1)
                logging.info(uv_trg.shape)
                cont_f0_lpf_trg = low_pass_filter(contf0_trg, int(1.0 / (5.0 * 0.001)), cutoff=20)
                logcontf0_trg = np.expand_dims(np.log(cont_f0_lpf_trg), axis=-1)
                logging.info(logcontf0_trg.shape)
                feat_trg = np.c_[uv_trg,logcontf0_trg,codeap_trg,mcep_trg]
                logging.info(feat_trg.shape)

                logging.info("generate")
                with torch.no_grad():
                    lat_src, _, _ = model_encoder(torch.FloatTensor(feat).cuda(), y_in_pp, clamp_vae=True, lat_dim=config.lat_dim)
                    lat_feat = sampling_vae_batch(lat_src.unsqueeze(0).repeat(args.n_smpl_dec,1,1), lat_dim=config.lat_dim)
                    lat_feat = torch.mean(lat_feat, 0)
                    lat_trg, _, _ = model_encoder(torch.FloatTensor(feat_trg).cuda(), y_in_pp, clamp_vae=True, lat_dim=config.lat_dim)
                    lat_feat_trg = sampling_vae_batch(lat_trg.unsqueeze(0).repeat(args.n_smpl_dec,1,1), lat_dim=config.lat_dim)
                    lat_feat_trg = torch.mean(lat_feat_trg, 0)
                    src_code = np.zeros((lat_feat.shape[0],2))
                    trg_code = np.zeros((lat_feat.shape[0],2))
                    trg_trg_code = np.zeros((lat_feat_trg.shape[0],2))
                    src_code[:,0] = 1
                    trg_code[:,1] = 1
                    trg_trg_code[:,1] = 1
                    src_code = torch.FloatTensor(src_code).cuda()
                    trg_code = torch.FloatTensor(trg_code).cuda()
                    trg_trg_code = torch.FloatTensor(trg_trg_code).cuda()
                    cvmcep, _, _ = model_decoder(torch.cat((trg_code, lat_feat),1), y_in_trg)
                    cvmcep = np.array(cvmcep.cpu().data.numpy(), dtype=np.float64)
                    cvmcep_src, _, _ = model_decoder(torch.cat((src_code, lat_feat),1), y_in_src)
                    cvmcep_src = np.array(cvmcep_src.cpu().data.numpy(), dtype=np.float64)
                    cvmcep_trg, _, _ = model_decoder(torch.cat((trg_trg_code, lat_feat_trg),1), y_in_trg)
                    cvmcep_trg = np.array(cvmcep_trg.cpu().data.numpy(), dtype=np.float64)
                logging.info(cvmcep.shape)
                logging.info(cvmcep_src.shape)
                logging.info(cvmcep_trg.shape)

                with torch.no_grad():
                    spcidx_src = torch.LongTensor(spcidx_src).cuda()
                    spcidx_trg = torch.LongTensor(spcidx_trg).cuda()

                trj_lat_src = np.array(torch.index_select(lat_src,0,spcidx_src).cpu().data.numpy(), dtype=np.float64)
                trj_lat_trg = np.array(torch.index_select(lat_trg,0,spcidx_trg).cpu().data.numpy(), dtype=np.float64)
                aligned_lat_srctrg, _, _, _ = dtw.dtw_org_to_trg(trj_lat_src, trj_lat_trg)
                lat_dist_srctrg = np.mean(np.sqrt(np.mean((aligned_lat_srctrg-trj_lat_trg)**2, axis=0)))
                _, _, lat_cdist_srctrg, _ = dtw.dtw_org_to_trg(trj_lat_trg, trj_lat_src, mcd=0)
                aligned_lat_trgsrc, _, _, _ = dtw.dtw_org_to_trg(trj_lat_trg, trj_lat_src)
                lat_dist_trgsrc = np.mean(np.sqrt(np.mean((aligned_lat_trgsrc-trj_lat_src)**2, axis=0)))
                _, _, lat_cdist_trgsrc, _ = dtw.dtw_org_to_trg(trj_lat_src, trj_lat_trg, mcd=0)
                logging.info("%lf %lf %lf %lf" % (lat_dist_srctrg, lat_cdist_srctrg, lat_dist_trgsrc, lat_cdist_trgsrc))
                lat_dist_rmse = (lat_dist_srctrg+lat_dist_trgsrc)/2
                lat_dist_cosim = (lat_cdist_srctrg+lat_cdist_trgsrc)/2
                lat_dist_rmse_enc_list.append(lat_dist_rmse)
                lat_dist_cosim_enc_list.append(lat_dist_cosim)
                logging.info("lat_dist_enc: %.6f %.6f" % (lat_dist_rmse, lat_dist_cosim))

                trj_lat_src = np.array(torch.index_select(lat_feat,0,spcidx_src).cpu().data.numpy(), dtype=np.float64)
                trj_lat_trg = np.array(torch.index_select(lat_feat_trg,0,spcidx_trg).cpu().data.numpy(), dtype=np.float64)
                aligned_lat_srctrg, _, _, _ = dtw.dtw_org_to_trg(trj_lat_src, trj_lat_trg)
                lat_dist_srctrg = np.mean(np.sqrt(np.mean((aligned_lat_srctrg-trj_lat_trg)**2, axis=0)))
                _, _, lat_cdist_srctrg, _ = dtw.dtw_org_to_trg(trj_lat_trg, trj_lat_src, mcd=0)
                aligned_lat_trgsrc, _, _, _ = dtw.dtw_org_to_trg(trj_lat_trg, trj_lat_src)
                lat_dist_trgsrc = np.mean(np.sqrt(np.mean((aligned_lat_trgsrc-trj_lat_src)**2, axis=0)))
                _, _, lat_cdist_trgsrc, _ = dtw.dtw_org_to_trg(trj_lat_src, trj_lat_trg, mcd=0)
                logging.info("%lf %lf %lf %lf" % (lat_dist_srctrg, lat_cdist_srctrg, lat_dist_trgsrc, lat_cdist_trgsrc))
                lat_dist_rmse = (lat_dist_srctrg+lat_dist_trgsrc)/2
                lat_dist_cosim = (lat_cdist_srctrg+lat_cdist_trgsrc)/2
                lat_dist_rmse_pri_list.append(lat_dist_rmse)
                lat_dist_cosim_pri_list.append(lat_dist_cosim)
                logging.info("lat_dist_pri: %.6f %.6f" % (lat_dist_rmse, lat_dist_cosim))

                spcidx_src = spcidx_src.cpu().data.numpy()
                _, _, _, mcdpow_arr = dtw.dtw_org_to_trg(np.array(cvmcep[np.array(spcidx_src),:], dtype=np.float64), np.array(mcepspc_trg[:,:], dtype=np.float64))
                _, _, _, mcd_arr = dtw.dtw_org_to_trg(np.array(cvmcep[np.array(spcidx_src),1:], dtype=np.float64), np.array(mcepspc_trg[:,1:], dtype=np.float64))
                mcdpow_mean = np.mean(mcdpow_arr)
                mcdpow_std = np.std(mcdpow_arr)
                mcd_mean = np.mean(mcd_arr)
                mcd_std = np.std(mcd_arr)
                logging.info("mcdpow_cv: %.6f dB +- %.6f" % (mcdpow_mean, mcdpow_std))
                logging.info("mcd_cv: %.6f dB +- %.6f" % (mcd_mean, mcd_std))
                mcdpow_cvlist.append(mcdpow_mean)
                mcdpowstd_cvlist.append(mcdpow_std)
                mcd_cvlist.append(mcd_mean)
                mcdstd_cvlist.append(mcd_std)
                cvlist.append(np.var(cvmcep[:,1:], axis=0))

                _, mcdpow_arr = dtw.calc_mcd(np.array(mcep[np.array(spcidx_src),:], dtype=np.float64), np.array(cvmcep_src[np.array(spcidx_src),:], dtype=np.float64))
                _, mcd_arr = dtw.calc_mcd(np.array(mcep[np.array(spcidx_src),1:], dtype=np.float64), np.array(cvmcep_src[np.array(spcidx_src),1:], dtype=np.float64))
                mcdpow_mean = np.mean(mcdpow_arr)
                mcdpow_std = np.std(mcdpow_arr)
                mcd_mean = np.mean(mcd_arr)
                mcd_std = np.std(mcd_arr)
                logging.info("mcdpow_src_cv: %.6f dB +- %.6f" % (mcdpow_mean, mcdpow_std))
                logging.info("mcd_src_cv: %.6f dB +- %.6f" % (mcd_mean, mcd_std))
                mcdpow_cvlist_src.append(mcdpow_mean)
                mcdpowstd_cvlist_src.append(mcdpow_std)
                mcd_cvlist_src.append(mcd_mean)
                mcdstd_cvlist_src.append(mcd_std)
                cvlist_src.append(np.var(cvmcep_src[:,1:], axis=0))

                spcidx_trg = spcidx_trg.cpu().data.numpy()
                _, mcdpow_arr = dtw.calc_mcd(np.array(mcepspc_trg[:,:], dtype=np.float64), np.array(cvmcep_trg[np.array(spcidx_trg),:], dtype=np.float64))
                _, mcd_arr = dtw.calc_mcd(np.array(mcepspc_trg[:,1:], dtype=np.float64), np.array(cvmcep_trg[np.array(spcidx_trg),1:], dtype=np.float64))
                mcdpow_mean = np.mean(mcdpow_arr)
                mcdpow_std = np.std(mcdpow_arr)
                mcd_mean = np.mean(mcd_arr)
                mcd_std = np.std(mcd_arr)
                logging.info("mcdpow_trg_cv: %.6f dB +- %.6f" % (mcdpow_mean, mcdpow_std))
                logging.info("mcd_trg_cv: %.6f dB +- %.6f" % (mcd_mean, mcd_std))
                mcdpow_cvlist_trg.append(mcdpow_mean)
                mcdpowstd_cvlist_trg.append(mcdpow_std)
                mcd_cvlist_trg.append(mcd_mean)
                mcdstd_cvlist_trg.append(mcd_std)
                cvlist_trg.append(np.var(cvmcep_trg[:,1:], axis=0))

                logging.info("mod_pow")
                cvmcep = mod_pow(cvmcep, mcep, alpha=args.mcep_alpha, irlen=IRLEN)
                logging.info(cvmcep.shape)

                logging.info("mod_pow_src")
                cvmcep_src = mod_pow(cvmcep_src, mcep, alpha=args.mcep_alpha, irlen=IRLEN)
                logging.info(cvmcep_src.shape)

                logging.info("mod_pow_trg")
                cvmcep_trg = mod_pow(cvmcep_trg, mcep_trg, alpha=args.mcep_alpha, irlen=IRLEN)
                logging.info(cvmcep_trg.shape)

                logging.info("cvmcep_gv")
                datamean = np.mean(cvmcep[:,1:], axis=0)
                cvmcep_gv =  np.c_[cvmcep[:,0], np.sqrt(gv_mean_trg/cvgv_mean) * (cvmcep[:,1:]-datamean) + datamean]
                logging.info(cvmcep_gv.shape)
                cvgvlist.append(np.var(cvmcep_gv[:,1:], axis=0))

                _, _, _, mcd_arr = dtw.dtw_org_to_trg(np.array(cvmcep_gv[np.array(spcidx_src),1:], dtype=np.float64), np.array(mcepspc_trg[:,1:], dtype=np.float64))
                mcd_mean = np.mean(mcd_arr)
                mcd_std = np.std(mcd_arr)
                logging.info("mcd_cvgv: %.6f dB +- %.6f" % (mcd_mean, mcd_std))
                mcd_cvgvlist.append(mcd_mean)
                mcdstd_cvgvlist.append(mcd_std)

                logging.info("mod_pow_gv")
                cvmcep_gv = mod_pow(cvmcep_gv, mcep, alpha=args.mcep_alpha, irlen=IRLEN)
                logging.info(cvmcep_gv.shape)

                logging.info("cvmcep_src_gv")
                datamean = np.mean(cvmcep_src[:,1:], axis=0)
                cvmcep_src_gv =  np.c_[cvmcep_src[:,0], np.sqrt(gv_mean_src/cvgvsrc_mean) * (cvmcep_src[:,1:]-datamean) + datamean]
                logging.info(cvmcep_src_gv.shape)
                cvgvlist_src.append(np.var(cvmcep_src_gv[:,1:], axis=0))

                _, mcd_arr = dtw.calc_mcd(np.array(mcep[np.array(spcidx_src),1:], dtype=np.float64), np.array(cvmcep_src_gv[np.array(spcidx_src),1:], dtype=np.float64))
                mcd_mean = np.mean(mcd_arr)
                mcd_std = np.std(mcd_arr)
                logging.info("mcd_src_cvgv: %.6f dB +- %.6f" % (mcd_mean, mcd_std))
                mcd_cvgvlist_src.append(mcd_mean)
                mcdstd_cvgvlist_src.append(mcd_std)

                logging.info("mod_pow_src_gv")
                cvmcep_src_gv = mod_pow(cvmcep_src_gv, mcep, alpha=args.mcep_alpha, irlen=IRLEN)
                logging.info(cvmcep_src_gv.shape)

                logging.info("cvmcep_trg_gv")
                datamean = np.mean(cvmcep_trg[:,1:], axis=0)
                cvmcep_trg_gv =  np.c_[cvmcep_trg[:,0], np.sqrt(gv_mean_trg/cvgvtrg_mean) * (cvmcep_trg[:,1:]-datamean) + datamean]
                logging.info(cvmcep_trg_gv.shape)
                cvgvlist_trg.append(np.var(cvmcep_trg_gv[:,1:], axis=0))

                _, mcd_arr = dtw.calc_mcd(np.array(mcepspc_trg[:,1:], dtype=np.float64), np.array(cvmcep_trg_gv[np.array(spcidx_trg),1:], dtype=np.float64))
                mcd_mean = np.mean(mcd_arr)
                mcd_std = np.std(mcd_arr)
                logging.info("mcd_trg_cvgv: %.6f dB +- %.6f" % (mcd_mean, mcd_std))
                mcd_cvgvlist_trg.append(mcd_mean)
                mcdstd_cvgvlist_trg.append(mcd_std)

                logging.info("mod_pow_trg_gv")
                cvmcep_trg_gv = mod_pow(cvmcep_trg_gv, mcep_trg, alpha=args.mcep_alpha, irlen=IRLEN)
                logging.info(cvmcep_trg_gv.shape)

                logging.info("mcdiff_nogv")
                mc_cv_diff_nogv = cvmcep-mcep
                logging.info(mc_cv_diff_nogv.shape)

                logging.info("mcdiff_gv")
                mc_cv_diff = cvmcep_gv-mcep
                logging.info(mc_cv_diff.shape)

                cvf0 = convert_f0(f0, f0_range_mean_src, f0_range_std_src, f0_range_mean_trg, f0_range_std_trg)

                logging.info("synth voco")
                cvsp = ps.mc2sp(cvmcep, args.mcep_alpha, fft_size)
                logging.info(cvsp.shape)
                wav = np.clip(pw.synthesize(cvf0, cvsp, ap, fs, frame_period=args.shiftms), -32768, 32767)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_noGV.wav"))
                wavfile.write(wavpath, fs, wav.astype(np.int16))
                logging.info(wavpath)

                logging.info("synth voco src")
                cvsp = ps.mc2sp(cvmcep_src, args.mcep_alpha, fft_size)
                logging.info(cvsp.shape)
                wav = np.clip(pw.synthesize(f0, cvsp, ap, fs, frame_period=args.shiftms), -32768, 32767)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_noGV_src.wav"))
                wavfile.write(wavpath, fs, wav.astype(np.int16))
                logging.info(wavpath)

                logging.info("synth voco trg")
                cvsp = ps.mc2sp(cvmcep_trg, args.mcep_alpha, fft_size)
                logging.info(cvsp.shape)
                wav = np.clip(pw.synthesize(f0_trg, cvsp, ap_trg, fs, frame_period=args.shiftms), -32768, 32767)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_noGV_trg.wav"))
                wavfile.write(wavpath, fs, wav.astype(np.int16))
                logging.info(wavpath)

                logging.info("synth vocoGV")
                cvsp = ps.mc2sp(cvmcep_gv, args.mcep_alpha, fft_size)
                logging.info(cvsp.shape)
                wav = np.clip(pw.synthesize(cvf0, cvsp, ap, fs, frame_period=args.shiftms), -32768, 32767)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_GV.wav"))
                wavfile.write(wavpath, fs, wav.astype(np.int16))
                logging.info(wavpath)

                logging.info("synth vocoGV src")
                cvsp = ps.mc2sp(cvmcep_src_gv, args.mcep_alpha, fft_size)
                logging.info(cvsp.shape)
                wav = np.clip(pw.synthesize(f0, cvsp, ap, fs, frame_period=args.shiftms), -32768, 32767)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_GV_src.wav"))
                wavfile.write(wavpath, fs, wav.astype(np.int16))
                logging.info(wavpath)

                logging.info("synth vocoGV trg")
                cvsp = ps.mc2sp(cvmcep_trg_gv, args.mcep_alpha, fft_size)
                logging.info(cvsp.shape)
                wav = np.clip(pw.synthesize(f0_trg, cvsp, ap_trg, fs, frame_period=args.shiftms), -32768, 32767)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_GV_trg.wav"))
                wavfile.write(wavpath, fs, wav.astype(np.int16))
                logging.info(wavpath)

                logging.info("synth diffGV")
                shiftl = int(fs/1000*args.shiftms)
                b = np.apply_along_axis(ps.mc2b, 1, mc_cv_diff, args.mcep_alpha)
                logging.info(b.shape)
                assert np.isfinite(b).all
                mlsa_fil = ps.synthesis.Synthesizer(MLSADF(mcep_dim, alpha=args.mcep_alpha), shiftl)
                wav = mlsa_fil.synthesis(x, b)
                wav = np.clip(wav, -32768, 32767)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_DiffGV.wav"))
                wavfile.write(wavpath, fs, wav.astype(np.int16))
                logging.info(wavpath)

                logging.info("synth diffGVF0")
                wav = low_cut_filter(wav, fs, 70)
                sp_diff = pw.cheaptrick(wav, f0, time_axis, fs, fft_size=fft_size)
                logging.info(sp_diff.shape)
                ap_diff = pw.d4c(wav, f0, time_axis, fs, fft_size=fft_size)
                logging.info(ap_diff.shape)
                wav = pw.synthesize(cvf0, sp_diff, ap_diff, fs, frame_period=args.shiftms)
                wav = np.clip(wav, -32768, 32767)
                wavpath = os.path.join(args.outdir, os.path.basename(feat_file).replace(".h5", "_DiffGVF0.wav"))
                wavfile.write(wavpath, fs, wav.astype(np.int16))
                logging.info(wavpath)


    with mp.Manager() as manager:
        logging.info("GRU-RNN decoding")
        processes = []
        cvlist = manager.list()
        mcd_cvlist = manager.list()
        mcdstd_cvlist = manager.list()
        mcdpow_cvlist = manager.list()
        mcdpowstd_cvlist = manager.list()
        cvlist_src = manager.list()
        mcd_cvlist_src = manager.list()
        mcdstd_cvlist_src = manager.list()
        mcdpow_cvlist_src = manager.list()
        mcdpowstd_cvlist_src = manager.list()
        cvlist_trg = manager.list()
        mcd_cvlist_trg = manager.list()
        mcdstd_cvlist_trg = manager.list()
        mcdpow_cvlist_trg = manager.list()
        mcdpowstd_cvlist_trg = manager.list()
        cvgvlist = manager.list()
        mcd_cvgvlist = manager.list()
        mcdstd_cvgvlist = manager.list()
        cvgvlist_src = manager.list()
        mcd_cvgvlist_src = manager.list()
        mcdstd_cvgvlist_src = manager.list()
        cvgvlist_trg = manager.list()
        mcd_cvgvlist_trg = manager.list()
        mcdstd_cvgvlist_trg = manager.list()
        difflist = manager.list()
        difflist = manager.list()
        mcd_difflist = manager.list()
        mcdstd_difflist = manager.list()
        diffgvlist = manager.list()
        mcd_diffgvlist = manager.list()
        mcdstd_diffgvlist = manager.list()
        lat_dist_rmse_enc_list = manager.list()
        lat_dist_cosim_enc_list = manager.list()
        lat_dist_rmse_pri_list = manager.list()
        lat_dist_cosim_pri_list = manager.list()
        gpu = 0
        for i, (feat_list, wav_list, wav_trg_list) in enumerate(zip(feat_lists, wav_lists, wav_trg_lists)):
            logging.info(i)
            p = mp.Process(target=decode_RNN, args=(feat_list, wav_list, wav_trg_list, gpu, cvlist, cvlist_src, cvgvlist, cvgvlist_src, difflist, diffgvlist, mcd_cvlist, mcdstd_cvlist, mcdpow_cvlist, mcdpowstd_cvlist, mcd_cvgvlist, mcdstd_cvgvlist, mcd_cvlist_src, mcdstd_cvlist_src, mcdpow_cvlist_src, mcdpowstd_cvlist_src, mcd_cvgvlist_src, mcdstd_cvgvlist_src, mcd_cvlist_trg, mcdstd_cvlist_trg, mcdpow_cvlist_trg, mcdpowstd_cvlist_trg, mcd_cvgvlist_trg, mcdstd_cvgvlist_trg, mcd_difflist, mcdstd_difflist, mcd_diffgvlist, mcdstd_diffgvlist, lat_dist_rmse_enc_list, lat_dist_cosim_enc_list, lat_dist_rmse_pri_list, lat_dist_cosim_pri_list,))
            p.start()
            processes.append(p)
            gpu += 1
            if (i + 1) % args.n_gpus == 0:
                gpu = 0

        # wait for all process
        for p in processes:
            p.join()

        # calculate cv_gv statistics
        #logging.info(gv_mean_trg)
        logging.info("mcdpow_cv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcdpow_cvlist)),np.std(np.array(mcdpow_cvlist)),np.mean(np.array(mcdpowstd_cvlist)),np.std(np.array(mcdpowstd_cvlist))))
        logging.info("mcd_cv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcd_cvlist)),np.std(np.array(mcd_cvlist)),np.mean(np.array(mcdstd_cvlist)),np.std(np.array(mcdstd_cvlist))))
        cvgv_ev_mean = np.mean(np.array(cvlist), axis=0)
        cvgv_ev_var = np.var(np.array(cvlist), axis=0)
        #logging.info(cvgv_ev_mean)
        logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_ev_mean)-np.log(gv_mean_trg)))), np.std(np.sqrt(np.square(np.log(cvgv_ev_mean)-np.log(gv_mean_trg))))))
        logging.info("mcd_cvGV: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcd_cvgvlist)),np.std(np.array(mcd_cvgvlist)),np.mean(np.array(mcdstd_cvgvlist)),np.std(np.array(mcdstd_cvgvlist))))
        cvgv_ev_mean = np.mean(np.array(cvgvlist), axis=0)
        cvgv_ev_var = np.var(np.array(cvgvlist), axis=0)
        #logging.info(cvgv_ev_mean)
        logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_ev_mean)-np.log(gv_mean_trg)))), np.std(np.sqrt(np.square(np.log(cvgv_ev_mean)-np.log(gv_mean_trg))))))
        #logging.info(gv_mean_src)
        logging.info("mcdpow_src_cv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcdpow_cvlist_src)),np.std(np.array(mcdpow_cvlist_src)),np.mean(np.array(mcdpowstd_cvlist_src)),np.std(np.array(mcdpowstd_cvlist_src))))
        logging.info("mcd_src_cv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcd_cvlist_src)),np.std(np.array(mcd_cvlist_src)),np.mean(np.array(mcdstd_cvlist_src)),np.std(np.array(mcdstd_cvlist_src))))
        cvgv_ev_mean = np.mean(np.array(cvlist_src), axis=0)
        cvgv_ev_var = np.var(np.array(cvlist_src), axis=0)
        #logging.info(cvgv_ev_mean)
        logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_ev_mean)-np.log(gv_mean_src)))), np.std(np.sqrt(np.square(np.log(cvgv_ev_mean)-np.log(gv_mean_src))))))
        logging.info("mcd_src_cvGV: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcd_cvgvlist_src)),np.std(np.array(mcd_cvgvlist_src)),np.mean(np.array(mcdstd_cvgvlist_src)),np.std(np.array(mcdstd_cvgvlist_src))))
        cvgv_ev_mean = np.mean(np.array(cvgvlist_src), axis=0)
        cvgv_ev_var = np.var(np.array(cvgvlist_src), axis=0)
        #logging.info(cvgv_ev_mean)
        logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_ev_mean)-np.log(gv_mean_src)))), np.std(np.sqrt(np.square(np.log(cvgv_ev_mean)-np.log(gv_mean_src))))))
        #logging.info(gv_mean_trg)
        logging.info("mcdpow_trg_cv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcdpow_cvlist_trg)),np.std(np.array(mcdpow_cvlist_trg)),np.mean(np.array(mcdpowstd_cvlist_trg)),np.std(np.array(mcdpowstd_cvlist_trg))))
        logging.info("mcd_trg_cv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcd_cvlist_trg)),np.std(np.array(mcd_cvlist_trg)),np.mean(np.array(mcdstd_cvlist_trg)),np.std(np.array(mcdstd_cvlist_trg))))
        cvgv_ev_mean = np.mean(np.array(cvlist_trg), axis=0)
        cvgv_ev_var = np.var(np.array(cvlist_trg), axis=0)
        #logging.info(cvgv_ev_mean)
        logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_ev_mean)-np.log(gv_mean_trg)))), np.std(np.sqrt(np.square(np.log(cvgv_ev_mean)-np.log(gv_mean_trg))))))
        logging.info("mcd_trg_cvGV: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcd_cvgvlist_trg)),np.std(np.array(mcd_cvgvlist_trg)),np.mean(np.array(mcdstd_cvgvlist_trg)),np.std(np.array(mcdstd_cvgvlist_trg))))
        cvgv_ev_mean = np.mean(np.array(cvgvlist_trg), axis=0)
        cvgv_ev_var = np.var(np.array(cvgvlist_trg), axis=0)
        #logging.info(cvgv_ev_mean)
        logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_ev_mean)-np.log(gv_mean_trg)))), np.std(np.sqrt(np.square(np.log(cvgv_ev_mean)-np.log(gv_mean_trg))))))
        logging.info("lat_dist_rmse_enc: %.6f (+- %.6f)" % (np.mean(np.array(lat_dist_rmse_enc_list)),np.std(np.array(lat_dist_rmse_enc_list))))
        logging.info("lat_dist_cosim_enc: %.6f (+- %.6f)" % (np.mean(np.array(lat_dist_cosim_enc_list)),np.std(np.array(lat_dist_cosim_enc_list))))
        logging.info("lat_dist_rmse_pri: %.6f (+- %.6f)" % (np.mean(np.array(lat_dist_rmse_pri_list)),np.std(np.array(lat_dist_rmse_pri_list))))
        logging.info("lat_dist_cosim_pri: %.6f (+- %.6f)" % (np.mean(np.array(lat_dist_cosim_pri_list)),np.std(np.array(lat_dist_cosim_pri_list))))


if __name__ == "__main__":
    main()
