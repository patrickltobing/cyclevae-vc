#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
# based on PyTorch implementation for WaveNet vocoder by Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import argparse
from distutils.util import strtobool
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.multiprocessing as mp

from utils import find_files, read_hdf5, read_txt, write_hdf5

from gru_vae import GRU_RNN, sampling_vae_batch

from dtw_c import dtw_c as dtw

np.set_printoptions(threshold=np.inf)


def main():
    parser = argparse.ArgumentParser()
    # decode setting
    parser.add_argument("--feats", required=True,
                        type=str, help="list or directory of source eval feat files")
    parser.add_argument("--feats_trg", required=True,
                        type=str, help="list or directory of source eval feat files")
    parser.add_argument("--stats_src", required=True,
                        type=str, help="hdf5 file including source statistics")
    parser.add_argument("--stats_trg", required=True,
                        type=str, help="hdf5 file including target statistics")
    parser.add_argument("--stats_jnt",
                        type=str, help="hdf5 file including target statistics")
    parser.add_argument("--model", required=True,
                        type=str, help="model file")
    parser.add_argument("--config", required=True,
                        type=str, help="configure file")
    parser.add_argument("--n_gpus", default=1,
                        type=int, help="number of gpus")
    parser.add_argument("--n_smpl_dec", default=300,
                        type=int, help="number of gpus")
    parser.add_argument("--outdir", required=True,
                        type=str, help="directory to save generated samples")
    parser.add_argument("--write_gv", default=False,
                        type=strtobool, help="flag to write gv stats")
    # other setting
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--GPU_device", default=0,
                        type=int, help="selection of GPU device")
    parser.add_argument("--verbose", default=1,
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

    # get source feat list
    if os.path.isdir(args.feats):
        feat_list = sorted(find_files(args.feats, "*.h5"))
    elif os.path.isfile(args.feats):
        feat_list = read_txt(args.feats)
    else:
        logging.error("--feats should be directory or list.")
        sys.exit(1)

    # get target feat list
    if os.path.isdir(args.feats_trg):
        feat_trg_list = sorted(find_files(args.feats_trg, "*.h5"))
    elif os.path.isfile(args.feats_trg):
        feat_trg_list = read_txt(args.feats_trg)
    else:
        logging.error("--feats_trg should be directory or list.")
        sys.exit(1)

    # prepare the file list for parallel decoding
    feat_lists = np.array_split(feat_list, args.n_gpus)
    feat_lists = [f_list.tolist() for f_list in feat_lists]
    feat_trg_lists = np.array_split(feat_trg_list, args.n_gpus)
    feat_trg_lists = [f_list.tolist() for f_list in feat_trg_lists]
    spk_src = os.path.basename(os.path.dirname(feat_lists[0][0]))
    spk_trg = os.path.basename(os.path.dirname(feat_trg_lists[0][0]))

    gv_mean_src = read_hdf5(args.stats_src, "/gv_range_mean")[1:]
    gv_mean_trg = read_hdf5(args.stats_trg, "/gv_range_mean")[1:]

    # define gpu decode function
    def gpu_decode(feat_list, feat_trg_list, gpu, cvlist=None, mcdlist=None, mcdstdlist=None, mcdpowlist=None, mcdpowstdlist=None, cvlist_src=None, mcdlist_src=None, mcdstdlist_src=None, mcdpowlist_src=None, mcdpowstdlist_src=None, cvlist_trg=None, mcdlist_trg=None, mcdstdlist_trg=None, mcdpowlist_trg=None, mcdpowstdlist_trg=None, lat_dist_rmse_enc_list=None, lat_dist_cosim_enc_list=None, lat_dist_rmse_pri_list=None, lat_dist_cosim_pri_list=None):
        with torch.cuda.device(gpu):
            mean_jnt = torch.FloatTensor(read_hdf5(args.stats_jnt, "/mean_feat_org_lf0_jnt")[config.stdim:]).cuda()
            std_jnt = torch.FloatTensor(read_hdf5(args.stats_jnt, "/scale_feat_org_lf0_jnt")[config.stdim:]).cuda()
            # define model and load parameters
            logging.info("model")
            logging.info(config)
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
                y_in_src = y_in_trg = torch.unsqueeze(torch.unsqueeze((0-mean_jnt)/std_jnt,0),0)
            for feat_file, feat_trg_file in zip(feat_list, feat_trg_list):
                # convert mcep
                logging.info("cvmcep " + feat_file + " " + feat_trg_file)

                feat = read_hdf5(feat_file, "/feat_org_lf0")
                feat_trg = read_hdf5(feat_trg_file, "/feat_org_lf0")
                logging.info(feat.shape)
                logging.info(feat_trg.shape)
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
                logging.info(cvmcep_trg.shape)
                cvlist.append(np.var(cvmcep[:,1:], axis=0))
                cvlist_src.append(np.var(cvmcep_src[:,1:], axis=0))
                cvlist_trg.append(np.var(cvmcep_trg[:,1:], axis=0))
                logging.info(len(cvlist))

                spcidx_src = read_hdf5(feat_file, "/spcidx_range")[0]
                mcep_trg = read_hdf5(feat_trg_file, "/mcepspc_range")
                _, _, _, mcdpow_arr = dtw.dtw_org_to_trg(np.array(cvmcep[np.array(spcidx_src),:], dtype=np.float64), np.array(mcep_trg[:,:], dtype=np.float64))
                _, _, _, mcd_arr = dtw.dtw_org_to_trg(np.array(cvmcep[np.array(spcidx_src),1:], dtype=np.float64), np.array(mcep_trg[:,1:], dtype=np.float64))
                mcdpow_mean = np.mean(mcdpow_arr)
                mcdpow_std = np.std(mcdpow_arr)
                mcd_mean = np.mean(mcd_arr)
                mcd_std = np.std(mcd_arr)
                logging.info("mcdpow: %.6f dB +- %.6f" % (mcdpow_mean, mcdpow_std))
                logging.info("mcd: %.6f dB +- %.6f" % (mcd_mean, mcd_std))
                mcdpowlist.append(mcdpow_mean)
                mcdpowstdlist.append(mcdpow_std)
                mcdlist.append(mcd_mean)
                mcdstdlist.append(mcd_std)
                
                mcep_src = read_hdf5(feat_file, "/mcepspc_range")
                _, mcdpow_arr = dtw.calc_mcd(np.array(mcep_src[:,:], dtype=np.float64), np.array(cvmcep_src[np.array(spcidx_src),:], dtype=np.float64))
                _, mcd_arr = dtw.calc_mcd(np.array(mcep_src[:,1:], dtype=np.float64), np.array(cvmcep_src[np.array(spcidx_src),1:], dtype=np.float64))
                mcdpow_mean = np.mean(mcdpow_arr)
                mcdpow_std = np.std(mcdpow_arr)
                mcd_mean = np.mean(mcd_arr)
                mcd_std = np.std(mcd_arr)
                logging.info("mcdpow_src: %.6f dB +- %.6f" % (mcdpow_mean, mcdpow_std))
                logging.info("mcd_src: %.6f dB +- %.6f" % (mcd_mean, mcd_std))
                mcdpowlist_src.append(mcdpow_mean)
                mcdpowstdlist_src.append(mcdpow_std)
                mcdlist_src.append(mcd_mean)
                mcdstdlist_src.append(mcd_std)

                spcidx_trg = read_hdf5(feat_trg_file, "/spcidx_range")[0]
                _, mcdpow_arr = dtw.calc_mcd(np.array(mcep_trg[:,:], dtype=np.float64), np.array(cvmcep_trg[np.array(spcidx_trg),:], dtype=np.float64))
                _, mcd_arr = dtw.calc_mcd(np.array(mcep_trg[:,1:], dtype=np.float64), np.array(cvmcep_trg[np.array(spcidx_trg),1:], dtype=np.float64))
                mcdpow_mean = np.mean(mcdpow_arr)
                mcdpow_std = np.std(mcdpow_arr)
                mcd_mean = np.mean(mcd_arr)
                mcd_std = np.std(mcd_arr)
                logging.info("mcdpow_trg: %.6f dB +- %.6f" % (mcdpow_mean, mcdpow_std))
                logging.info("mcd_trg: %.6f dB +- %.6f" % (mcd_mean, mcd_std))
                mcdpowlist_trg.append(mcdpow_mean)
                mcdpowstdlist_trg.append(mcdpow_std)
                mcdlist_trg.append(mcd_mean)
                mcdstdlist_trg.append(mcd_std)

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
                
    # parallel decode training
    with mp.Manager() as manager:
        gpu = 0
        processes = []
        cvlist = manager.list()
        mcdlist = manager.list()
        mcdstdlist = manager.list()
        mcdpowlist = manager.list()
        mcdpowstdlist = manager.list()
        cvlist_src = manager.list()
        mcdlist_src = manager.list()
        mcdstdlist_src = manager.list()
        mcdpowlist_src = manager.list()
        mcdpowstdlist_src = manager.list()
        cvlist_trg = manager.list()
        mcdlist_trg = manager.list()
        mcdstdlist_trg = manager.list()
        mcdpowlist_trg = manager.list()
        mcdpowstdlist_trg = manager.list()
        lat_dist_rmse_enc_list = manager.list()
        lat_dist_cosim_enc_list = manager.list()
        lat_dist_rmse_pri_list = manager.list()
        lat_dist_cosim_pri_list = manager.list()
        for i, (feat_list, feat_trg_list) in enumerate(zip(feat_lists, feat_trg_lists)):
            logging.info(i)
            p = mp.Process(target=gpu_decode, args=(feat_list, feat_trg_list, gpu, cvlist, mcdlist, mcdstdlist, mcdpowlist, mcdpowstdlist, cvlist_src, mcdlist_src, mcdstdlist_src, mcdpowlist_src, mcdpowstdlist_src, cvlist_trg, mcdlist_trg, mcdstdlist_trg, mcdpowlist_trg, mcdpowstdlist_trg, lat_dist_rmse_enc_list, lat_dist_cosim_enc_list, lat_dist_rmse_pri_list, lat_dist_cosim_pri_list,))
            p.start()
            processes.append(p)
            gpu += 1
            if (i + 1) % args.n_gpus == 0:
                gpu = 0
        # wait for all process
        for p in processes:
            p.join()
        # calculate cv_gv statistics
        cvgv_mean = np.mean(np.array(cvlist), axis=0)
        cvgv_var = np.var(np.array(cvlist), axis=0)
        cvgvsrc_mean = np.mean(np.array(cvlist_src), axis=0)
        cvgvsrc_var = np.var(np.array(cvlist_src), axis=0)
        cvgvtrg_mean = np.mean(np.array(cvlist_trg), axis=0)
        cvgvtrg_var = np.var(np.array(cvlist_trg), axis=0)
        logging.info(args.stats_src)
        logging.info(args.stats_trg)
        #logging.info(gv_mean_trg)
        logging.info("mcdpow: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcdpowlist)),np.std(np.array(mcdpowlist)),np.mean(np.array(mcdpowstdlist)),np.std(np.array(mcdpowstdlist))))
        logging.info("mcd: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcdlist)),np.std(np.array(mcdlist)),np.mean(np.array(mcdstdlist)),np.std(np.array(mcdstdlist))))
        #logging.info(cvgv_mean)
        logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean_trg)))), np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean_trg))))))
        logging.info("mcdpow_src: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcdpowlist_src)),np.std(np.array(mcdpowlist_src)),np.mean(np.array(mcdpowstdlist_src)),np.std(np.array(mcdpowstdlist_src))))
        logging.info("mcd_src: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcdlist_src)),np.std(np.array(mcdlist_src)),np.mean(np.array(mcdstdlist_src)),np.std(np.array(mcdstdlist_src))))
        #logging.info(cvgvsrc_mean)
        logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgvsrc_mean)-np.log(gv_mean_src)))), np.std(np.sqrt(np.square(np.log(cvgvsrc_mean)-np.log(gv_mean_src))))))
        logging.info("mcdpow_trg: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcdpowlist_trg)),np.std(np.array(mcdpowlist_trg)),np.mean(np.array(mcdpowstdlist_trg)),np.std(np.array(mcdpowstdlist_trg))))
        logging.info("mcd_trg: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcdlist_trg)),np.std(np.array(mcdlist_trg)),np.mean(np.array(mcdstdlist_trg)),np.std(np.array(mcdstdlist_trg))))
        #logging.info(cvgvtrg_mean)
        logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgvtrg_mean)-np.log(gv_mean_trg)))), np.std(np.sqrt(np.square(np.log(cvgvtrg_mean)-np.log(gv_mean_trg))))))
        logging.info("lat_dist_rmse_enc: %.6f (+- %.6f)" % (np.mean(np.array(lat_dist_rmse_enc_list)),np.std(np.array(lat_dist_rmse_enc_list))))
        logging.info("lat_dist_cosim_enc: %.6f (+- %.6f)" % (np.mean(np.array(lat_dist_cosim_enc_list)),np.std(np.array(lat_dist_cosim_enc_list))))
        logging.info("lat_dist_rmse_pri: %.6f (+- %.6f)" % (np.mean(np.array(lat_dist_rmse_pri_list)),np.std(np.array(lat_dist_rmse_pri_list))))
        logging.info("lat_dist_cosim_pri: %.6f (+- %.6f)" % (np.mean(np.array(lat_dist_cosim_pri_list)),np.std(np.array(lat_dist_cosim_pri_list))))
        if args.write_gv:
            tmp_str = args.model.split('/')[1].split('_')
            mdl_name = tmp_str[2]+"_"+tmp_str[3]
            logging.info(mdl_name)
            string_path = mdl_name+"-"+str(config.n_cyc)+"-"+str(config.lat_dim)+"-"+str(torch.load(args.model)["iterations"])+"-"+spk_trg+"-"+str(args.n_smpl_dec)
            logging.info(string_path)
            string_mean = "/cvgv_mean_"+string_path
            string_var = "/cvgv_var_"+string_path
            write_hdf5(args.stats_src, string_mean, cvgv_mean)
            write_hdf5(args.stats_src, string_var, cvgv_var)
            string_mean = "/cvgvsrc_mean_"+string_path
            string_var = "/cvgvsrc_var_"+string_path
            write_hdf5(args.stats_src, string_mean, cvgvsrc_mean)
            write_hdf5(args.stats_src, string_var, cvgvsrc_var)
            string_mean = "/cvgvtrg_mean_"+string_path
            string_var = "/cvgvtrg_var_"+string_path
            write_hdf5(args.stats_src, string_mean, cvgvtrg_mean)
            write_hdf5(args.stats_src, string_var, cvgvtrg_var)


if __name__ == "__main__":
    main()
