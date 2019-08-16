#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
# based on PyTorch implementation for WaveNet vocoder by Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import argparse
from dateutil.relativedelta import relativedelta
from distutils.util import strtobool
import logging
import itertools
import os
import sys
import time

import numpy as np
import six
import torch
from torch.autograd import Variable
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import find_files
from utils import read_hdf5
from utils import read_txt

from gru_vae import initialize
from gru_vae import GRU_RNN
from gru_vae import TWFSEloss
from gru_vae import sampling_vae_batch, loss_vae

from dataset import FeatureDatasetSingleVAE, padding

from dtw_c import dtw_c as dtw

#np.set_printoptions(threshold=np.inf)
#torch.set_printoptions(threshold=np.inf)


def train_generator(dataloader, device, batch_size=80):
    while True:
        c_idx = 0
        # process over all of files
        for idx, batch in enumerate(dataloader):
            flens = batch['flen_src'].data.numpy()
            max_flen = np.max(flens) ## get max frame length
            flens_spc_src = batch['flen_spc_src'].data.numpy()
            max_flen_spc_src = np.max(flens_spc_src) ## get max frame length
            flens_src_trg = batch['flen_src_trg'].data.numpy()
            max_flen_src_trg = np.max(flens_src_trg) ## get max frame length
            flens_spc_src_trg = batch['flen_spc_src_trg'].data.numpy()
            max_flen_spc_src_trg = np.max(flens_spc_src_trg) ## get max frame length
            hs_src = batch['h_src'][:,:max_flen].to(device)
            src_codes = batch['src_code'][:,:max_flen].to(device)
            trg_codes = batch['trg_code'][:,:max_flen].to(device)
            cvs_src = batch['cv_src'][:,:max_flen].to(device)
            spcidcs_src = batch['spcidx_src'][:,:max_flen_spc_src].to(device)
            hs_src_trg = batch['h_src_trg'][:,:max_flen_src_trg].to(device)
            spcidcs_src_trg = batch['spcidx_src_trg'][:,:max_flen_spc_src_trg].to(device)
            featfiles_src = batch['featfile_src']
            featfiles_src_trg = batch['featfile_src_trg']
            n_batch_utt = hs_src.size(0)

            # use mini batch
            if batch_size != 0:
                src_idx_s = 0
                src_idx_e = batch_size-1
                spcidcs_src_s_idx = np.repeat(-1,n_batch_utt)
                spcidcs_src_e_idx = np.repeat(-1,n_batch_utt)
                s_flag = np.repeat(False,n_batch_utt)
                e_flag = np.repeat(True,n_batch_utt)
                flen_acc = np.repeat(batch_size,n_batch_utt)
                for j in range(n_batch_utt):
                    for i in range(spcidcs_src_e_idx[j]+1,flens_spc_src[j]):
                        if not s_flag[j] and spcidcs_src[j,i] >= src_idx_s:
                            if spcidcs_src[j,i] > src_idx_e:
                                spcidcs_src_s_idx[j] = -1
                                break
                            spcidcs_src_s_idx[j] = i
                            s_flag[j] = True
                            e_flag[j] = False
                            if i == flens_spc_src[j]-1:
                                spcidcs_src_e_idx[j] = i
                                s_flag[j] = False
                                e_flag[j] = True
                                break
                        elif not e_flag[j] and (spcidcs_src[j,i] >= src_idx_e or i == flens_spc_src[j]-1):
                            if spcidcs_src[j,i] > src_idx_e:
                                spcidcs_src_e_idx[j] = i-1
                            else:
                                spcidcs_src_e_idx[j] = i
                            s_flag[j] = False
                            e_flag[j] = True
                            break
                select_utt_idx = [i for i in range(n_batch_utt)]
                yield hs_src, src_codes[:,src_idx_s:src_idx_e+1], trg_codes[:,src_idx_s:src_idx_e+1], hs_src_trg, cvs_src, src_idx_s, src_idx_e, spcidcs_src_s_idx, spcidcs_src_e_idx, c_idx, idx, spcidcs_src, spcidcs_src_trg, featfiles_src, featfiles_src_trg, flens, flens_src_trg, flens_spc_src, flens_spc_src_trg, select_utt_idx, flen_acc, n_batch_utt
                while src_idx_e < max_flen-1:
                    src_idx_s = src_idx_e + 1
                    src_idx_e = src_idx_s+batch_size-1
                    if src_idx_e >= max_flen:
                        src_idx_e = max_flen-1
                    select_utt_idx  = []
                    for j in range(n_batch_utt):
                        if spcidcs_src_e_idx[j] < flens_spc_src[j]-1:
                            if src_idx_e >= flens[j]:
                                flen_acc[j] = flens[j]-src_idx_s
                            for i in range(spcidcs_src_e_idx[j]+1,flens_spc_src[j]):
                                if not s_flag[j] and spcidcs_src[j,i] >= src_idx_s:
                                    if spcidcs_src[j,i] > src_idx_e:
                                        spcidcs_src_s_idx[j] = -1
                                        break
                                    spcidcs_src_s_idx[j] = i
                                    s_flag[j] = True
                                    e_flag[j] = False
                                    if i == flens_spc_src[j]-1:
                                        spcidcs_src_e_idx[j] = i
                                        s_flag[j] = False
                                        e_flag[j] = True
                                        break
                                elif not e_flag[j] and (spcidcs_src[j,i] >= src_idx_e or i == flens_spc_src[j]-1):
                                    if spcidcs_src[j,i] > src_idx_e:
                                        spcidcs_src_e_idx[j] = i-1
                                    else:
                                        spcidcs_src_e_idx[j] = i
                                    s_flag[j] = False
                                    e_flag[j] = True
                                    break
                            select_utt_idx.append(j)
                    yield hs_src, src_codes[:,src_idx_s:src_idx_e+1], trg_codes[:,src_idx_s:src_idx_e+1], hs_src_trg, cvs_src, src_idx_s, src_idx_e, spcidcs_src_s_idx, spcidcs_src_e_idx, c_idx, idx, spcidcs_src, spcidcs_src_trg, featfiles_src, featfiles_src_trg, flens, flens_src_trg, flens_spc_src, flens_spc_src_trg, select_utt_idx, flen_acc, n_batch_utt

            # use utterance batch
            else:
                yield hs_src, src_codes, trg_codes, hs_src_trg, cvs_src, c_idx, idx, spcidcs_src, spcidcs_src_trg, featfiles_src, featfiles_src_trg, flens, flens_src_trg, flens_spc_src, flens_spc_src_trg, n_batch_utt

            c_idx += 1
            if c_idx > 0:
            #if c_idx > 1:
            #if c_idx > 2:
                break

        if batch_size > 0:
            yield [], [], [], [], [], [], [], [], [], -1, -1, [], [], [], [], [], [], [], [], [], [], []
        else:
            yield [], [], [], [], [], -1, -1, [], [], [], [], [], [], [], [], []


def save_checkpoint(checkpoint_dir, model_encoder, model_decoder, optimizer, numpy_random_state, torch_random_state, iterations):
    model_encoder.cpu()
    model_decoder.cpu()
    checkpoint = {
        "model_encoder": model_encoder.state_dict(),
        "model_decoder": model_decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "numpy_random_state": numpy_random_state,
        "torch_random_state": torch_random_state,
        "iterations": iterations}
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(checkpoint, checkpoint_dir + "/checkpoint-%d.pkl" % iterations)
    model_encoder.cuda()
    model_decoder.cuda()
    logging.info("%d-iter checkpoint created." % iterations)


def main():
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument("--feats_src", required=True,
                        type=str, help="directory or list of source feat files")
    parser.add_argument("--feats_src_trg", required=True,
                        type=str, help="directory or list of source feat files")
    parser.add_argument("--feats_trg", required=True,
                        type=str, help="directory or list of target feat files")
    parser.add_argument("--feats_trg_src", required=True,
                        type=str, help="directory or list of target feat files")
    parser.add_argument("--feats_eval_src", required=True,
                        type=str, help="directory or list of evaluation source feat files")
    parser.add_argument("--feats_eval_trg", required=True,
                        type=str, help="directory or list of evaluation target feat files")
    parser.add_argument("--stats_src", required=True,
                        type=str, help="hdf5 file including source statistics")
    parser.add_argument("--stats_trg", required=True,
                        type=str, help="hdf5 file including target statistics")
    parser.add_argument("--stats_jnt", required=True,
                        type=str, help="hdf5 file including target statistics")
    parser.add_argument("--spk_src", required=True,
                        type=str, help="hdf5 file including source statistics")
    parser.add_argument("--spk_trg", required=True,
                        type=str, help="hdf5 file including source statistics")
    parser.add_argument("--expdir", required=True,
                        type=str, help="directory to save the model")
    parser.add_argument('--batch_size_utt', default=8, type=int,
                        help='Batch size')
    parser.add_argument('--batch_size_utt_eval', default=8, type=int,
                        help='Batch size')
    parser.add_argument('--pad_len', default=2200, type=int,
                        help='Batch length')
    parser.add_argument('--n_workers', default=2, type=int,
                        help='# of workers for dataset')
    parser.add_argument('--stdim', default=4, type=int,
                        help='stdim for mcep')
    # network structure setting
    parser.add_argument("--in_dim", default=54,
                        type=int, help="number of dimension of input features")
    parser.add_argument("--lat_dim", default=32,
                        type=int, help="number of dimension of output features")
    parser.add_argument("--out_dim", default=50,
                        type=int, help="number of dimension of output features")
    parser.add_argument("--hidden_layers", default=1,
                        type=int, help="number of hidden layers")
    parser.add_argument("--hidden_units", default=1024,
                        type=int, help="number of hidden units")
    parser.add_argument("--kernel_size", default=3,
                        type=int, help="number of hidden units")
    parser.add_argument("--dilation_size", default=2,
                        type=int, help="number of hidden units")
    parser.add_argument("--n_cyc", default=2,
                        type=int, help="number of hidden units")
    parser.add_argument("--do_prob", default=0.5,
                        type=float, help="dropout probability")
    # network training setting
    parser.add_argument("--lr", default=1e-4,
                        type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="weight decay coefficient")
    parser.add_argument("--batch_size", default=80,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--epoch_count", default=400,
                        type=int, help="number of training epochs")
    # other setting
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--resume", default=None,
                        type=str, help="model path to restart training")
    parser.add_argument("--GPU_device", default=0,
                        type=int, help="selection of GPU device")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log level")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]  = str(args.GPU_device)

    # make experimental directory
    if not os.path.exists(args.expdir):
        os.makedirs(args.expdir)

    # set log level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/train.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/train.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/train.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    # fix seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == "cpu":
        raise ValueError('ERROR: Training by CPU is not acceptable.')

    if args.n_cyc < 1:
        half_cyc = True
        args.n_cyc = 1
    else:
        half_cyc = False

    # save args as conf
    torch.save(args, args.expdir + "/model.conf")

    stdim = args.stdim
    stdim_ = stdim+1

    # define statistics src
    mean_jnt = torch.FloatTensor(read_hdf5(args.stats_jnt, "/mean_feat_org_lf0_jnt"))
    std_jnt = torch.FloatTensor(read_hdf5(args.stats_jnt, "/scale_feat_org_lf0_jnt"))
    mean_jnt_trg = torch.FloatTensor(read_hdf5(args.stats_jnt, "/mean_feat_org_lf0_jnt")[stdim:])
    std_jnt_trg = torch.FloatTensor(read_hdf5(args.stats_jnt, "/scale_feat_org_lf0_jnt")[stdim:])

    if torch.cuda.is_available():
        mean_jnt = mean_jnt.cuda()
        std_jnt = std_jnt.cuda()
        mean_jnt_trg = mean_jnt_trg.cuda()
        std_jnt_trg = std_jnt_trg.cuda()
    else:
        logging.error("gpu is not available. please check the setting.")

    # define network
    model_encoder = GRU_RNN(
        in_dim=args.in_dim,
        out_dim=args.lat_dim*2,
        hidden_layers=args.hidden_layers,
        hidden_units=args.hidden_units,
        kernel_size=args.kernel_size,
        dilation_size=args.dilation_size,
        do_prob=args.do_prob,
        scale_out_flag=False)
    logging.info(model_encoder)
    model_decoder = GRU_RNN(
        in_dim=args.lat_dim+2,
        out_dim=args.out_dim,
        hidden_layers=args.hidden_layers,
        hidden_units=args.hidden_units,
        kernel_size=args.kernel_size,
        dilation_size=args.dilation_size,
        do_prob=args.do_prob,
        scale_in_flag=False)
    logging.info(model_decoder)
    criterion_mcd = TWFSEloss()

    # send to gpu
    if torch.cuda.is_available():
        model_encoder.cuda()
        model_decoder.cuda()
        criterion_mcd.cuda()
    else:
        logging.error("gpu is not available. please check the setting.")
        sys.exit(1)
    model_encoder.apply(initialize)
    model_encoder.train()
    model_decoder.apply(initialize)
    model_decoder.train()
    model_encoder.scale_in.weight = torch.nn.Parameter(torch.diag(1.0/std_jnt.data).unsqueeze(2))
    model_encoder.scale_in.bias = torch.nn.Parameter(-(mean_jnt.data/std_jnt.data))
    model_decoder.scale_out.weight = torch.nn.Parameter(torch.diag(std_jnt_trg.data).unsqueeze(2))
    model_decoder.scale_out.bias = torch.nn.Parameter(mean_jnt_trg.data)
    if args.resume is None:
        epoch_idx = 0
    else:
        checkpoint = torch.load(args.resume)
        model_encoder.load_state_dict(checkpoint["model_encoder"])
        model_decoder.load_state_dict(checkpoint["model_decoder"])
        epoch_idx = checkpoint["iterations"]
        logging.info("restored from %d-iter checkpoint." % (epoch_idx))

    init_pp = np.zeros((args.batch_size_utt,1,args.lat_dim*2))
    y_in_pp = torch.FloatTensor(init_pp).cuda()
    y_in_src = y_in_trg = torch.unsqueeze(torch.unsqueeze((0-mean_jnt_trg)/std_jnt_trg,0),0).repeat(args.batch_size_utt,1,1)
    with torch.no_grad():
        init_pp_eval = np.zeros((args.batch_size_utt_eval,1,args.lat_dim*2))
        y_in_pp_eval = torch.FloatTensor(init_pp_eval).cuda()
        y_in_src_eval = y_in_trg_eval = torch.unsqueeze(torch.unsqueeze((0-mean_jnt_trg)/std_jnt_trg,0),0).repeat(args.batch_size_utt_eval,1,1)

    for param in model_encoder.parameters():
        param.requires_grad = True
    for param in model_decoder.parameters():
        param.requires_grad = True
    for param in model_encoder.scale_in.parameters():
        param.requires_grad = False
    for param in model_decoder.scale_out.parameters():
        param.requires_grad = False
    module_list = list(model_encoder.conv.parameters())
    module_list += list(model_encoder.gru.parameters()) + list(model_encoder.out_1.parameters())
    module_list += list(model_decoder.conv.parameters())
    module_list += list(model_decoder.gru.parameters()) + list(model_decoder.out_1.parameters())
    optimizer = torch.optim.Adam(module_list, lr=args.lr)
    if args.resume is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    parameters = filter(lambda p: p.requires_grad, model_encoder.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (encoder): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_decoder.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (decoder): %.3f million' % parameters)

    # define generator training
    if os.path.isdir(args.feats_src):
        feat_list_src = sorted(find_files(args.feats_src, "*.h5", use_dir_name=False))
    elif os.path.isfile(args.feats_src):
        feat_list_src = read_txt(args.feats_src)
    else:
        logging.error("--feats_src should be directory or list.")
        sys.exit(1)
    if os.path.isdir(args.feats_src_trg):
        feat_list_src_trg = sorted(find_files(args.feats_src_trg, "*.h5", use_dir_name=False))
    elif os.path.isfile(args.feats_src_trg):
        feat_list_src_trg = read_txt(args.feats_src_trg)
    else:
        logging.error("--feats_src_trg should be directory or list.")
        sys.exit(1)
    assert(len(feat_list_src) == len(feat_list_src_trg))
    logging.info("number of training src data = %d." % len(feat_list_src))
    if os.path.isdir(args.feats_trg):
        feat_list_trg = sorted(find_files(args.feats_trg, "*.h5", use_dir_name=False))
    elif os.path.isfile(args.feats_trg):
        feat_list_trg = read_txt(args.feats_trg)
    else:
        logging.error("--feats_trg should be directory or list.")
        sys.exit(1)
    if os.path.isdir(args.feats_trg_src):
        feat_list_trg_src = sorted(find_files(args.feats_trg_src, "*.h5", use_dir_name=False))
    elif os.path.isfile(args.feats_trg_src):
        feat_list_trg_src = read_txt(args.feats_trg_src)
    else:
        logging.error("--feats_trg_src should be directory or list.")
        sys.exit(1)
    assert(len(feat_list_trg) == len(feat_list_trg_src))
    logging.info("number of training trg data = %d." % len(feat_list_trg))

    n_train_data = len(feat_list_src) + len(feat_list_trg)
    mod_train_batch = n_train_data % args.batch_size_utt
    if mod_train_batch > 0:
        init_pp_mod = np.zeros((mod_train_batch,1,args.lat_dim*2))
        y_in_pp_mod= torch.FloatTensor(init_pp_mod).cuda()
        y_in_src_mod = y_in_trg_mod = torch.unsqueeze(torch.unsqueeze((0-mean_jnt_trg)/std_jnt_trg,0),0).repeat(mod_train_batch,1,1)

    # define generator evaluation
    if os.path.isdir(args.feats_eval_src):
        feat_list_eval_src = sorted(find_files(args.feats_eval_src, "*.h5", use_dir_name=False))
    elif os.path.isfile(args.feats_eval_src):
        feat_list_eval_src = read_txt(args.feats_eval_src)
    else:
        logging.error("--feats_eval_src should be directory or list.")
        sys.exit(1)
    if os.path.isdir(args.feats_eval_trg):
        feat_list_eval_trg = sorted(find_files(args.feats_eval_trg, "*.h5", use_dir_name=False))
    elif os.path.isfile(args.feats_eval_trg):
        feat_list_eval_trg = read_txt(args.feats_eval_trg)
    else:
        logging.error("--feats_eval_trg should be directory or list.")
        sys.exit(1)
    assert(len(feat_list_eval_src) == len(feat_list_eval_trg))
    logging.info("number of evaluation data = %d." % len(feat_list_eval_src))

    n_eval_data = len(feat_list_eval_src)
    mod_eval_batch = n_eval_data % args.batch_size_utt_eval
    if mod_eval_batch > 0:
        with torch.no_grad():
            init_pp_eval_mod = np.zeros((mod_eval_batch,1,args.lat_dim*2))
            y_in_pp_eval_mod = torch.FloatTensor(init_pp_eval_mod).cuda()
            y_in_src_eval_mod = y_in_trg_eval_mod = torch.unsqueeze(torch.unsqueeze((0-mean_jnt_trg)/std_jnt_trg,0),0).repeat(mod_eval_batch,1,1)

    # data
    def zero_pad(x): return padding(x, args.pad_len, value=0.0)
    pad_transform = transforms.Compose([zero_pad])
    dataset = FeatureDatasetSingleVAE(feat_list_src+feat_list_trg, feat_list_src_trg+feat_list_trg_src, pad_transform, args.spk_src)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_utt, shuffle=True, num_workers=args.n_workers)
    dataset_eval_src = FeatureDatasetSingleVAE(feat_list_eval_src, feat_list_eval_trg, pad_transform, args.spk_src)
    dataloader_eval_src = DataLoader(dataset_eval_src, batch_size=args.batch_size_utt_eval, num_workers=args.n_workers)
    dataset_eval_trg = FeatureDatasetSingleVAE(feat_list_eval_trg, feat_list_eval_src, pad_transform, args.spk_src)
    dataloader_eval_trg = DataLoader(dataset_eval_trg, batch_size=args.batch_size_utt_eval, num_workers=args.n_workers)

    # generator optimization instance
    generator_src = train_generator(dataloader, device, batch_size=args.batch_size)

    # generator eval instance
    generator_eval_src = train_generator(dataloader_eval_src, device, batch_size=0)
    generator_eval_trg = train_generator(dataloader_eval_trg, device, batch_size=0)

    gv_trg_mean = read_hdf5(args.stats_trg, "/gv_range_mean")[1:]
    gv_src_mean = read_hdf5(args.stats_src, "/gv_range_mean")[1:]

    # train
    batch_lat_src = [None]*args.n_cyc
    y_in_pp_src = [None]*args.n_cyc
    h_in_pp_src = [None]*args.n_cyc
    batch_trj_src_src = [None]*args.n_cyc
    y_in_src_src = [None]*args.n_cyc
    h_in_src_src = [None]*args.n_cyc
    batch_trj_src_trg = [None]*args.n_cyc
    y_in_src_trg = [None]*args.n_cyc
    h_in_src_trg = [None]*args.n_cyc
    batch_lat_src_trg = [None]*args.n_cyc
    y_in_pp_src_trg = [None]*args.n_cyc
    h_in_pp_src_trg = [None]*args.n_cyc
    batch_trj_src_trg_src = [None]*args.n_cyc
    y_in_src_trg_src = [None]*args.n_cyc
    h_in_src_trg_src = [None]*args.n_cyc
    batch_lat_trg_ = [None]*args.n_cyc
    batch_trj_trg_trg_ = [None]*args.n_cyc
    batch_trj_trg_src_ = [None]*args.n_cyc
    batch_lat_trg_src_ = [None]*args.n_cyc
    batch_trj_trg_src_trg_ = [None]*args.n_cyc
    batch_lat_src_ = [None]*args.n_cyc
    batch_trj_src_src_ = [None]*args.n_cyc
    batch_trj_src_trg_ = [None]*args.n_cyc
    batch_lat_src_trg_ = [None]*args.n_cyc
    batch_trj_src_trg_src_ = [None]*args.n_cyc
    batch_loss_mcd_trg_trg = [None]*args.n_cyc
    batch_loss_mcd_trg_src_trg = [None]*args.n_cyc
    batch_loss_mcd_trg_src = [None]*args.n_cyc
    batch_loss_mcd_src_src = [None]*args.n_cyc
    batch_loss_mcd_src_trg_src = [None]*args.n_cyc
    batch_loss_mcd_src_trg = [None]*args.n_cyc
    batch_loss_lat_src = [None]*args.n_cyc
    batch_loss_lat_trg = [None]*args.n_cyc
    batch_loss_lat_src_cv = [None]*args.n_cyc
    batch_loss_lat_trg_cv = [None]*args.n_cyc
    batch_gv_trg_trg = [None]*args.n_cyc
    batch_mcdpow_trg_trg = [None]*args.n_cyc
    batch_mcd_trg_trg = [None]*args.n_cyc
    batch_gv_trg_src_trg = [None]*args.n_cyc
    batch_mcdpow_trg_src_trg = [None]*args.n_cyc
    batch_mcd_trg_src_trg = [None]*args.n_cyc
    batch_gv_trg_src = [None]*args.n_cyc
    batch_mcdpow_trg_src = [None]*args.n_cyc
    batch_mcd_trg_src = [None]*args.n_cyc
    batch_lat_dist_trgsrc1 = [None]*args.n_cyc
    batch_lat_dist_trgsrc2 = [None]*args.n_cyc
    batch_lat_cdist_trgsrc1 = [None]*args.n_cyc
    batch_lat_cdist_trgsrc2 = [None]*args.n_cyc
    batch_gv_src_src = [None]*args.n_cyc
    batch_mcdpow_src_src = [None]*args.n_cyc
    batch_mcd_src_src = [None]*args.n_cyc
    batch_gv_src_trg_src = [None]*args.n_cyc
    batch_mcdpow_src_trg_src = [None]*args.n_cyc
    batch_mcd_src_trg_src = [None]*args.n_cyc
    batch_gv_src_trg = [None]*args.n_cyc
    batch_mcdpow_src_trg = [None]*args.n_cyc
    batch_mcd_src_trg = [None]*args.n_cyc
    batch_lat_dist_srctrg1 = [None]*args.n_cyc
    batch_lat_dist_srctrg2 = [None]*args.n_cyc
    batch_lat_cdist_srctrg1 = [None]*args.n_cyc
    batch_lat_cdist_srctrg2 = [None]*args.n_cyc
    loss = []
    loss_mcd_trg_trg = []
    loss_mcd_trg_src_trg = []
    loss_mcd_trg_src = []
    loss_mcd_src_src = []
    loss_mcd_src_trg_src = []
    loss_mcd_src_trg = []
    loss_lat_src = []
    loss_lat_trg = []
    loss_lat_src_cv = []
    loss_lat_trg_cv = []
    gv_trg_trg = []
    mcdpow_trg_trg = []
    mcd_trg_trg = []
    gv_trg_src_trg = []
    mcdpow_trg_src_trg = []
    mcd_trg_src_trg = []
    gv_trg_src = []
    mcdpow_trg_src = []
    mcd_trg_src = []
    lat_dist_trgsrc1 = []
    lat_dist_trgsrc2 = []
    gv_src_src = []
    mcdpow_src_src = []
    mcd_src_src = []
    gv_src_trg_src = []
    mcdpow_src_trg_src = []
    mcd_src_trg_src = []
    gv_src_trg = []
    mcdpow_src_trg = []
    mcd_src_trg = []
    lat_dist_srctrg1 = []
    lat_dist_srctrg2 = []
    for i in range(args.n_cyc):
        loss_mcd_trg_trg.append([])
        loss_mcd_trg_src_trg.append([])
        loss_mcd_trg_src.append([])
        loss_mcd_src_src.append([])
        loss_mcd_src_trg_src.append([])
        loss_mcd_src_trg.append([])
        loss_lat_src.append([])
        loss_lat_trg.append([])
        loss_lat_src_cv.append([])
        loss_lat_trg_cv.append([])
        gv_trg_trg.append([])
        mcdpow_trg_trg.append([])
        mcd_trg_trg.append([])
        gv_trg_src_trg.append([])
        mcdpow_trg_src_trg.append([])
        mcd_trg_src_trg.append([])
        gv_trg_src.append([])
        mcdpow_trg_src.append([])
        mcd_trg_src.append([])
        lat_dist_trgsrc1.append([])
        lat_dist_trgsrc2.append([])
        gv_src_src.append([])
        mcdpow_src_src.append([])
        mcd_src_src.append([])
        gv_src_trg_src.append([])
        mcdpow_src_trg_src.append([])
        mcd_src_trg_src.append([])
        gv_src_trg.append([])
        mcdpow_src_trg.append([])
        mcd_src_trg.append([])
        lat_dist_srctrg1.append([])
        lat_dist_srctrg2.append([])
    total = []
    n_ev_cyc = 1
    #if args.n_cyc > 1:    
    #    n_ev_cyc = 2
    #else:
    #    n_ev_cyc = 1
    eval_loss_mcd_trg_trg = [None]*n_ev_cyc
    eval_loss_mcd_trg_src_trg = [None]*n_ev_cyc
    eval_loss_mcd_trg_src = [None]*n_ev_cyc
    eval_loss_mcd_src_src = [None]*n_ev_cyc
    eval_loss_mcd_src_trg_src = [None]*n_ev_cyc
    eval_loss_mcd_src_trg = [None]*n_ev_cyc
    eval_loss_lat_src = [None]*n_ev_cyc
    eval_loss_lat_trg = [None]*n_ev_cyc
    eval_loss_lat_src_cv = [None]*n_ev_cyc
    eval_loss_lat_trg_cv = [None]*n_ev_cyc
    eval_gv_trg_trg = [None]*n_ev_cyc
    eval_mcdpow_trg_trg = [None]*n_ev_cyc
    eval_mcd_trg_trg = [None]*n_ev_cyc
    eval_gv_trg_src_trg = [None]*n_ev_cyc
    eval_mcdpow_trg_src_trg = [None]*n_ev_cyc
    eval_mcd_trg_src_trg = [None]*n_ev_cyc
    eval_gv_trg_src = [None]*n_ev_cyc
    eval_mcdpow_trg_src = [None]*n_ev_cyc
    eval_mcdpowstd_trg_src = [None]*n_ev_cyc
    eval_mcd_trg_src = [None]*n_ev_cyc
    eval_mcdstd_trg_src = [None]*n_ev_cyc
    eval_lat_dist_trgsrc1 = [None]*n_ev_cyc
    eval_lat_dist_trgsrc2 = [None]*n_ev_cyc
    eval_gv_src_src = [None]*n_ev_cyc
    eval_mcdpow_src_src = [None]*n_ev_cyc
    eval_mcd_src_src = [None]*n_ev_cyc
    eval_gv_src_trg_src = [None]*n_ev_cyc
    eval_mcdpow_src_trg_src = [None]*n_ev_cyc
    eval_mcd_src_trg_src = [None]*n_ev_cyc
    eval_gv_src_trg = [None]*n_ev_cyc
    eval_mcdpow_src_trg = [None]*n_ev_cyc
    eval_mcdpowstd_src_trg = [None]*n_ev_cyc
    eval_mcd_src_trg = [None]*n_ev_cyc
    eval_mcdstd_src_trg = [None]*n_ev_cyc
    eval_lat_dist_srctrg1 = [None]*n_ev_cyc
    eval_lat_dist_srctrg2 = [None]*n_ev_cyc
    prev_featfile_src = np.repeat("",args.batch_size_utt)
    iter_idx = 0 
    iter_count = 0 
    min_idx = -1
    min_eval_mcdpow_src_trg = 99999999.99
    min_eval_mcdpowstd_src_trg = 99999999.99
    min_eval_mcd_src_trg = 99999999.99
    min_eval_mcdstd_src_trg = 99999999.99
    if args.resume is not None:
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
    logging.info("==%d EPOCH==" % (epoch_idx+1))
    logging.info("Training data")
    while epoch_idx < args.epoch_count:
        start = time.time()
        if args.batch_size > 0:
            if iter_count > 0:
                featfile_src_ = featfile_src
                featfile_src_trg_ = featfile_src_trg
                spcidx_src_ = spcidx_src
                prev_flens_src = flens_src
                flens_spc_src_ = flens_spc_src
                batch_src_trg_ = batch_src_trg
                spcidx_src_trg_ = spcidx_src_trg
                flens_spc_src_trg_ = flens_spc_src_trg
                n_batch_utt_ = n_batch_utt
            batch_src, batch_src_src_code, batch_src_trg_code, batch_src_trg, batch_cv_src, src_idx_s, src_idx_e, spcidx_src_s_idx, spcidx_src_e_idx, c_idx_src, utt_idx_src, spcidx_src, spcidx_src_trg, featfile_src, featfile_src_trg, flens_src, flens_src_trg, flens_spc_src, flens_spc_src_trg, select_utt_idx, flen_acc, n_batch_utt = next(generator_src)
            if iter_count > 0 and (src_idx_s == 0 or c_idx_src < 0):
                with torch.no_grad():
                    if n_batch_utt_ == args.batch_size_utt:
                        trj_lat_srctrg, _, _ = model_encoder(batch_src_trg_, y_in_pp, clamp_vae=True, lat_dim=args.lat_dim)
                    else:
                        trj_lat_srctrg, _, _ = model_encoder(batch_src_trg_, y_in_pp_mod, clamp_vae=True, lat_dim=args.lat_dim)
                for i in range(n_batch_utt_):
                    _, _, batch_mcdpow_src_trg[0], _ = dtw.dtw_org_to_trg(np.array(torch.index_select(trj_src_trg[i],0,spcidx_src_[i,:flens_spc_src_[i]]).cpu().data.numpy(), dtype=np.float64), np.array(torch.index_select(batch_src_trg_[i][:,stdim:],0,spcidx_src_trg_[i,:flens_spc_src_trg_[i]]).cpu().data.numpy(), dtype=np.float64))
                    _, _, batch_mcd_src_trg[0], _ = dtw.dtw_org_to_trg(np.array(torch.index_select(trj_src_trg[i][:,1:],0,spcidx_src_[i,:flens_spc_src_[i]]).cpu().data.numpy(), dtype=np.float64), np.array(torch.index_select(batch_src_trg_[i][:,stdim_:],0,spcidx_src_trg_[i,:flens_spc_src_trg_[i]]).cpu().data.numpy(), dtype=np.float64))
                    trj_lat_srctrg_ = np.array(torch.index_select(trj_lat_srctrg[i],0,spcidx_src_trg_[i,:flens_spc_src_trg_[i]]).cpu().data.numpy(), dtype=np.float64)
                    trj_lat_src_ = np.array(torch.index_select(trj_lat_src[i],0,spcidx_src_[i,:flens_spc_src_[i]]).cpu().data.numpy(), dtype=np.float64)
                    aligned_lat_srctrg1, _, _, _ = dtw.dtw_org_to_trg(trj_lat_src_, trj_lat_srctrg_)
                    batch_lat_dist_srctrg1[0] = np.mean(np.sqrt(np.mean((aligned_lat_srctrg1-trj_lat_srctrg_)**2, axis=0)))
                    _, _, batch_lat_cdist_srctrg1[0], _ = dtw.dtw_org_to_trg(trj_lat_srctrg_, trj_lat_src_, mcd=0)
                    aligned_lat_srctrg2, _, _, _ = dtw.dtw_org_to_trg(trj_lat_srctrg_, trj_lat_src_)
                    batch_lat_dist_srctrg2[0] = np.mean(np.sqrt(np.mean((aligned_lat_srctrg2-trj_lat_src_)**2, axis=0)))
                    _, _, batch_lat_cdist_srctrg2[0], _ = dtw.dtw_org_to_trg(trj_lat_src_, trj_lat_srctrg_, mcd=0)
                    if os.path.basename(os.path.dirname(featfile_src_[i])) == args.spk_src:
                        mcdpow_src_trg[0].append(batch_mcdpow_src_trg[0])
                        mcd_src_trg[0].append(batch_mcd_src_trg[0])
                        batch_lat_dist_srctrg1[0] = (batch_lat_dist_srctrg1[0]+batch_lat_dist_srctrg2[0])/2
                        lat_dist_srctrg1[0].append(batch_lat_dist_srctrg1[0])
                        batch_lat_dist_srctrg2[0] = (batch_lat_cdist_srctrg1[0]+batch_lat_cdist_srctrg2[0])/2
                        lat_dist_srctrg2[0].append(batch_lat_dist_srctrg2[0])
                        logging.info("batch srctrg loss %s %s = %.3f dB %.3f dB , %.3f %.3f" % (featfile_src_[i], featfile_src_trg_[i], batch_mcdpow_src_trg[0], batch_mcd_src_trg[0], batch_lat_dist_srctrg1[0], batch_lat_dist_srctrg2[0]))
                    else:
                        mcdpow_trg_src[0].append(batch_mcdpow_src_trg[0])
                        mcd_trg_src[0].append(batch_mcd_src_trg[0])
                        batch_lat_dist_trgsrc1[0] = (batch_lat_dist_srctrg1[0]+batch_lat_dist_srctrg2[0])/2
                        lat_dist_trgsrc1[0].append(batch_lat_dist_trgsrc1[0])
                        batch_lat_dist_trgsrc2[0] = (batch_lat_cdist_srctrg1[0]+batch_lat_cdist_srctrg2[0])/2
                        lat_dist_trgsrc2[0].append(batch_lat_dist_trgsrc2[0])
                        logging.info("batch trgsrc loss %s %s = %.3f dB %.3f dB , %.3f %.3f" % (featfile_src_[i], featfile_src_trg_[i], batch_mcdpow_src_trg[0], batch_mcd_src_trg[0], batch_lat_dist_trgsrc1[0], batch_lat_dist_trgsrc2[0]))
        else:
            batch_src, batch_src_src_code, batch_src_trg_code, batch_src_trg, batch_cv_src, c_idx_src, utt_idx_src, spcidx_src, spcidx_src_trg, featfile_src, featfile_src_trg, flens_src, flens_src_trg, flens_spc_src, flens_spc_src_trg, n_batch_utt = next(generator_src)
        if c_idx_src < 0:
            numpy_random_state = np.random.get_state()
            torch_random_state = torch.get_rng_state()
            # save current epoch model
            save_checkpoint(args.expdir, model_encoder, model_decoder, optimizer, numpy_random_state, torch_random_state, epoch_idx + 1)
            if args.batch_size > 0:
                batch_src, batch_src_src_code, batch_src_trg_code, batch_src_trg, batch_cv_src, src_idx_s, src_idx_e, spcidx_src_s_idx, spcidx_src_e_idx, c_idx_src, utt_idx_src, spcidx_src, spcidx_src_trg, featfile_src, featfile_src_trg, flens_src, flens_src_trg, flens_spc_src, flens_spc_src_trg, select_utt_idx, flen_acc, n_batch_utt = next(generator_src)
            else:
                batch_src, batch_src_src_code, batch_src_trg_code, batch_src_trg, batch_cv_src, c_idx_src, utt_idx_src, spcidx_src, spcidx_src_trg, featfile_src, featfile_src_trg, flens_src, flens_src_trg, flens_spc_src, flens_spc_src_trg, n_batch_utt = next(generator_src)
            numpy_random_state = np.random.get_state()
            torch_random_state = torch.get_rng_state()
            # report current epoch
            text_log = "%.3f ;; " % np.mean(loss)
            #for i in range(args.n_cyc):
            for i in range(n_ev_cyc):
                eval_gv_trg_trg[i] = np.mean(np.sqrt(np.square(np.log(np.mean(gv_trg_trg[i], axis=0))-np.log(gv_trg_mean))))
                eval_gv_src_trg[i] = np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_trg[i], axis=0))-np.log(gv_trg_mean))))
                eval_gv_trg_src_trg[i] = np.mean(np.sqrt(np.square(np.log(np.mean(gv_trg_src_trg[i], axis=0))-np.log(gv_trg_mean))))
                eval_gv_src_src[i] = np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_src[i], axis=0))-np.log(gv_src_mean))))
                eval_gv_trg_src[i] = np.mean(np.sqrt(np.square(np.log(np.mean(gv_trg_src[i], axis=0))-np.log(gv_src_mean))))
                eval_gv_src_trg_src[i] = np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_trg_src[i], axis=0))-np.log(gv_src_mean))))
                text_log += "[%d] %.3f %.3f %.3f %.3f %.3f %.3f ; %.3f %.3f %.3f %.3f ; %.6f %.3f dB %.6f dB , %.3f %.3f dB %.3f dB , %.6f %.3f dB (+- %.3f) %.6f dB (+- %.3f) , %.6f %.6f ; %.6f %.3f dB %.6f dB , %.3f %.3f dB %.3f dB , %.6f %.3f dB (+- %.3f) %.6f dB (+- %.3f) , %.6f %.6f ;; " % (
                             i+1, np.mean(loss_mcd_trg_trg[i]), np.mean(loss_mcd_trg_src_trg[i]), np.mean(loss_mcd_trg_src[i]),
                                 np.mean(loss_mcd_src_src[i]), np.mean(loss_mcd_src_trg_src[i]), np.mean(loss_mcd_src_trg[i]),
                                     np.mean(loss_lat_trg[i]), np.mean(loss_lat_trg_cv[i]), np.mean(loss_lat_src[i]), np.mean(loss_lat_src_cv[i]),
                                         eval_gv_trg_trg[i], np.mean(mcdpow_trg_trg[i]), np.mean(mcd_trg_trg[i]),
                                             eval_gv_trg_src_trg[i], np.mean(mcdpow_trg_src_trg[i]), np.mean(mcd_trg_src_trg[i]),
                                                 eval_gv_trg_src[i], np.mean(mcdpow_trg_src[i]), np.std(mcdpow_trg_src[i]), np.mean(mcd_trg_src[i]), np.std(mcd_trg_src[i]),
                                                     np.mean(lat_dist_trgsrc1[i]), np.mean(lat_dist_trgsrc2[i]), eval_gv_src_src[i], np.mean(mcdpow_src_src[i]), np.mean(mcd_src_src[i]),
                                                         eval_gv_src_trg_src[i], np.mean(mcdpow_src_trg_src[i]), np.mean(mcd_src_trg_src[i]),
                                                             eval_gv_src_trg[i], np.mean(mcdpow_src_trg[i]), np.std(mcdpow_src_trg[i]), np.mean(mcd_src_trg[i]), np.std(mcd_src_trg[i]),
                                                                 np.mean(lat_dist_srctrg1[i]), np.mean(lat_dist_srctrg2[i]))
            logging.info("(EPOCH:%d) average optimization loss = %s  (%.3f min., %.3f sec / batch)" % (epoch_idx + 1, text_log, np.sum(total) / 60.0, np.mean(total)))
            logging.info("estimated training required time = {0.days:02}:{0.hours:02}:{0.minutes:02}:{0.seconds:02}".format(relativedelta(seconds=int((args.epoch_count - (epoch_idx + 1)) * np.sum(total)))))
            model_encoder.eval()
            model_decoder.eval()
            for param in model_encoder.parameters():
                param.requires_grad = False
            for param in model_decoder.parameters():
                param.requires_grad = False
            # compute loss in evaluation data
            loss = []
            loss_mcd_trg_trg = []
            loss_mcd_trg_src_trg = []
            loss_mcd_trg_src = []
            loss_mcd_src_src = []
            loss_mcd_src_trg_src = []
            loss_mcd_src_trg = []
            loss_lat_src = []
            loss_lat_trg = []
            loss_lat_src_cv = []
            loss_lat_trg_cv = []
            gv_trg_trg = []
            mcdpow_trg_trg = []
            mcd_trg_trg = []
            gv_trg_src_trg = []
            mcdpow_trg_src_trg = []
            mcd_trg_src_trg = []
            gv_trg_src = []
            mcdpow_trg_src = []
            mcd_trg_src = []
            lat_dist_trgsrc1 = []
            lat_dist_trgsrc2 = []
            gv_src_src = []
            mcdpow_src_src = []
            mcd_src_src = []
            gv_src_trg_src = []
            mcdpow_src_trg_src = []
            mcd_src_trg_src = []
            gv_src_trg = []
            mcdpow_src_trg = []
            mcd_src_trg = []
            lat_dist_srctrg1 = []
            lat_dist_srctrg2 = []
            for i in range(n_ev_cyc):
                loss_mcd_trg_trg.append([])
                loss_mcd_trg_src_trg.append([])
                loss_mcd_trg_src.append([])
                loss_mcd_src_src.append([])
                loss_mcd_src_trg_src.append([])
                loss_mcd_src_trg.append([])
                loss_lat_src.append([])
                loss_lat_trg.append([])
                loss_lat_src_cv.append([])
                loss_lat_trg_cv.append([])
                gv_trg_trg.append([])
                mcdpow_trg_trg.append([])
                mcd_trg_trg.append([])
                gv_trg_src_trg.append([])
                mcdpow_trg_src_trg.append([])
                mcd_trg_src_trg.append([])
                gv_trg_src.append([])
                mcdpow_trg_src.append([])
                mcd_trg_src.append([])
                lat_dist_trgsrc1.append([])
                lat_dist_trgsrc2.append([])
                gv_src_src.append([])
                mcdpow_src_src.append([])
                mcd_src_src.append([])
                gv_src_trg_src.append([])
                mcdpow_src_trg_src.append([])
                mcd_src_trg_src.append([])
                gv_src_trg.append([])
                mcdpow_src_trg.append([])
                mcd_src_trg.append([])
                lat_dist_srctrg1.append([])
                lat_dist_srctrg2.append([])
            total = []
            iter_count = 0
            logging.info("Evaluation data")
            with torch.no_grad():
                while True:
                    start = time.time()
                    batch_src_, batch_src_src_code_, batch_src_trg_code_, batch_src_trg_, batch_cv_src_, c_idx, utt_idx, spcidx_src_, spcidx_src_trg_, featfile_src_, featfile_src_trg_, flens_src_, flens_src_trg_, flens_spc_src_, flens_spc_src_trg_, n_batch_utt_ = next(generator_eval_src)
                    batch_trg_, batch_trg_trg_code_, batch_trg_src_code_, batch_trg_src_, batch_cv_trg_, c_idx, utt_idx, spcidx_trg_, spcidx_trg_src_, featfile_trg_, featfile_trg_src_, flens_trg_, flens_trg_src_, flens_spc_trg_, flens_spc_trg_src_, n_batch_utt_ = next(generator_eval_trg)
                    if c_idx < 0:
                        break
                    for i in range(n_batch_utt_):
                        logging.info("%s %s %d %d %d %d" % (featfile_src_[i], featfile_src_trg_[i], flens_src_[i], flens_src_trg_[i], flens_spc_src_[i], flens_spc_src_trg_[i]))
                        logging.info("%s %s %d %d %d %d" % (featfile_trg_[i], featfile_trg_src_[i], flens_trg_[i], flens_trg_src_[i], flens_spc_trg_[i], flens_spc_trg_src_[i]))

                    if n_batch_utt_ == args.batch_size_utt_eval:
                        y_in_pp_eval_ = y_in_pp_eval
                        y_in_trg_eval_ = y_in_trg_eval
                        y_in_src_eval_ = y_in_src_eval
                    else:
                        y_in_pp_eval_ = y_in_pp_eval_mod
                        y_in_trg_eval_ = y_in_trg_eval_mod
                        y_in_src_eval_ = y_in_src_eval_mod

                    trj_lat_srctrg, _, _ = model_encoder(batch_src_trg_, y_in_pp_eval_, clamp_vae=True, lat_dim=args.lat_dim)
                    trj_lat_trgsrc, _, _ = model_encoder(batch_trg_src_, y_in_pp_eval_, clamp_vae=True, lat_dim=args.lat_dim)
                    for i in range(n_ev_cyc):
                        batch_mcdpow_src_src[i] = []
                        batch_mcd_src_src[i] = []
                        batch_mcdpow_src_trg_src[i] = []
                        batch_mcd_src_trg_src[i] = []
                        batch_mcdpow_src_trg[i] = []
                        batch_mcd_src_trg[i] = []
                        batch_mcdpow_trg_trg[i] = []
                        batch_mcd_trg_trg[i] = []
                        batch_mcdpow_trg_src_trg[i] = []
                        batch_mcd_trg_src_trg[i] = []
                        batch_mcdpow_trg_src[i] = []
                        batch_mcd_trg_src[i] = []
                        batch_lat_dist_srctrg1[i] = []
                        batch_lat_dist_srctrg2[i] = []
                        batch_lat_dist_trgsrc1[i] = []
                        batch_lat_dist_trgsrc2[i] = []
                        if i > 0:
                            batch_lat_trg_[i], _, _ = model_encoder(torch.cat((batch_trg_[:,:,:stdim], batch_trj_trg_src_trg_[i-1]),2), y_in_pp_eval_, clamp_vae=True, lat_dim=args.lat_dim)
                            batch_lat_src_[i], _, _ = model_encoder(torch.cat((batch_src_[:,:,:stdim], batch_trj_src_trg_src_[i-1]),2), y_in_pp_eval_, clamp_vae=True, lat_dim=args.lat_dim)

                            batch_trj_trg_trg_[i], _, _ = model_decoder(torch.cat((batch_trg_trg_code_, sampling_vae_batch(batch_lat_trg_[i], lat_dim=args.lat_dim)),2), y_in_trg_eval_)
                            batch_trj_trg_src_[i], _, _ = model_decoder(torch.cat((batch_trg_src_code_, sampling_vae_batch(batch_lat_trg_[i], lat_dim=args.lat_dim)),2), y_in_src_eval_)

                            batch_trj_src_src_[i], _, _ = model_decoder(torch.cat((batch_src_src_code_, sampling_vae_batch(batch_lat_src_[i], lat_dim=args.lat_dim)),2), y_in_src_eval_)
                            batch_trj_src_trg_[i], _, _ = model_decoder(torch.cat((batch_src_trg_code_, sampling_vae_batch(batch_lat_src_[i], lat_dim=args.lat_dim)),2), y_in_trg_eval_)

                            batch_lat_trg_src_[i], _, _ = model_encoder(torch.cat((batch_cv_trg_, batch_trj_trg_src_[i]),2), y_in_pp_eval_, clamp_vae=True, lat_dim=args.lat_dim)
                            batch_lat_src_trg_[i], _, _ = model_encoder(torch.cat((batch_cv_src_, batch_trj_src_trg_[i]),2), y_in_pp_eval_, clamp_vae=True, lat_dim=args.lat_dim)

                            batch_trj_trg_src_trg_[i], _, _ = model_decoder(torch.cat((batch_trg_trg_code_, sampling_vae_batch(batch_lat_trg_src_[i], lat_dim=args.lat_dim)),2), y_in_trg_eval_)
                            batch_trj_src_trg_src_[i], _, _ = model_decoder(torch.cat((batch_src_src_code_, sampling_vae_batch(batch_lat_src_trg_[i], lat_dim=args.lat_dim)),2), y_in_src_eval_)
                        else:
                            batch_lat_trg_[0], _, _ = model_encoder(batch_trg_, y_in_pp_eval_, clamp_vae=True, lat_dim=args.lat_dim)
                            batch_lat_src_[0], _, _ = model_encoder(batch_src_, y_in_pp_eval_, clamp_vae=True, lat_dim=args.lat_dim)

                            batch_trj_trg_trg_[0], _, _ = model_decoder(torch.cat((batch_trg_trg_code_, sampling_vae_batch(batch_lat_trg_[0], lat_dim=args.lat_dim)),2), y_in_trg_eval_)
                            batch_trj_trg_src_[0], _, _ = model_decoder(torch.cat((batch_trg_src_code_, sampling_vae_batch(batch_lat_trg_[0], lat_dim=args.lat_dim)),2), y_in_src_eval_)

                            batch_trj_src_src_[0], _, _ = model_decoder(torch.cat((batch_src_src_code_, sampling_vae_batch(batch_lat_src_[0], lat_dim=args.lat_dim)),2), y_in_src_eval_)
                            batch_trj_src_trg_[0], _, _ = model_decoder(torch.cat((batch_src_trg_code_, sampling_vae_batch(batch_lat_src_[0], lat_dim=args.lat_dim)),2), y_in_trg_eval_)

                            batch_lat_trg_src_[0], _, _ = model_encoder(torch.cat((batch_cv_trg_, batch_trj_trg_src_[0]),2), y_in_pp_eval_, clamp_vae=True, lat_dim=args.lat_dim)
                            batch_lat_src_trg_[0], _, _ = model_encoder(torch.cat((batch_cv_src_, batch_trj_src_trg_[0]),2), y_in_pp_eval_, clamp_vae=True, lat_dim=args.lat_dim)

                            batch_trj_trg_src_trg_[0], _, _ = model_decoder(torch.cat((batch_trg_trg_code_, sampling_vae_batch(batch_lat_trg_src_[0], lat_dim=args.lat_dim)),2), y_in_trg_eval_)
                            batch_trj_src_trg_src_[0], _, _ = model_decoder(torch.cat((batch_src_src_code_, sampling_vae_batch(batch_lat_src_trg_[0], lat_dim=args.lat_dim)),2), y_in_src_eval_)

                            for j in range(n_batch_utt_):
                                gv_src_src[i].append(np.var(batch_trj_src_src_[i][j,:flens_src_[j],1:].cpu().data.numpy(), axis=0))
                                gv_src_trg[i].append(np.var(batch_trj_src_trg_[i][j,:flens_src_[j],1:].cpu().data.numpy(), axis=0))
                                gv_src_trg_src[i].append(np.var(batch_trj_src_trg_src_[i][j,:flens_src_[j],1:].cpu().data.numpy(), axis=0))
                                gv_trg_trg[i].append(np.var(batch_trj_trg_trg_[i][j,:flens_trg_[j],1:].cpu().data.numpy(), axis=0))
                                gv_trg_src[i].append(np.var(batch_trj_trg_src_[i][j,:flens_trg_[j],1:].cpu().data.numpy(), axis=0))
                                gv_trg_src_trg[i].append(np.var(batch_trj_trg_src_trg_[i][j,:flens_trg_[j],1:].cpu().data.numpy(), axis=0))

                                trj_lat_srctrg_ = np.array(torch.index_select(trj_lat_srctrg[j],0,spcidx_src_trg_[j,:flens_spc_src_trg_[j]]).cpu().data.numpy(), dtype=np.float64)
                                trj_lat_src_ = np.array(torch.index_select(batch_lat_src_[0][j],0,spcidx_src_[j,:flens_spc_src_[j]]).cpu().data.numpy(), dtype=np.float64)
                                aligned_lat_srctrg1, _, _, _ = dtw.dtw_org_to_trg(trj_lat_src_, trj_lat_srctrg_)
                                tmp_batch_lat_dist_srctrg1 = np.mean(np.sqrt(np.mean((aligned_lat_srctrg1-trj_lat_srctrg_)**2, axis=0)))
                                _, _, tmp_batch_lat_cdist_srctrg1, _ = dtw.dtw_org_to_trg(trj_lat_srctrg_, trj_lat_src_, mcd=0)
                                aligned_lat_srctrg2, _, _, _ = dtw.dtw_org_to_trg(trj_lat_srctrg_, trj_lat_src_)
                                tmp_batch_lat_dist_srctrg2 = np.mean(np.sqrt(np.mean((aligned_lat_srctrg2-trj_lat_src_)**2, axis=0)))
                                _, _, tmp_batch_lat_cdist_srctrg2, _ = dtw.dtw_org_to_trg(trj_lat_src_, trj_lat_srctrg_, mcd=0)

                                tmp_batch_lat_dist_srctrg1 = (tmp_batch_lat_dist_srctrg1+tmp_batch_lat_dist_srctrg2)/2
                                lat_dist_srctrg1[0].append(tmp_batch_lat_dist_srctrg1)
                                tmp_batch_lat_dist_srctrg2 = (tmp_batch_lat_cdist_srctrg1+tmp_batch_lat_cdist_srctrg2)/2
                                lat_dist_srctrg2[0].append(tmp_batch_lat_dist_srctrg2)

                                batch_lat_dist_srctrg1[0].append(tmp_batch_lat_dist_srctrg1)
                                batch_lat_dist_srctrg2[0].append(tmp_batch_lat_dist_srctrg2)

                                trj_lat_trgsrc_ = np.array(torch.index_select(trj_lat_trgsrc[j],0,spcidx_trg_src_[j,:flens_spc_trg_src_[j]]).cpu().data.numpy(), dtype=np.float64)
                                trj_lat_trg_ = np.array(torch.index_select(batch_lat_trg_[0][j],0,spcidx_trg_[j,:flens_spc_trg_[j]]).cpu().data.numpy(), dtype=np.float64)
                                aligned_lat_trgsrc1, _, _, _ = dtw.dtw_org_to_trg(trj_lat_trg_, trj_lat_trgsrc_)
                                tmp_batch_lat_dist_trgsrc1 = np.mean(np.sqrt(np.mean((aligned_lat_trgsrc1-trj_lat_trgsrc_)**2, axis=0)))
                                _, _, tmp_batch_lat_cdist_trgsrc1, _ = dtw.dtw_org_to_trg(trj_lat_trgsrc_, trj_lat_trg_, mcd=0)
                                aligned_lat_trgsrc2, _, _, _ = dtw.dtw_org_to_trg(trj_lat_trgsrc_, trj_lat_trg_)
                                tmp_batch_lat_dist_trgsrc2 = np.mean(np.sqrt(np.mean((aligned_lat_trgsrc2-trj_lat_trg_)**2, axis=0)))
                                _, _, tmp_batch_lat_cdist_trgsrc2, _ = dtw.dtw_org_to_trg(trj_lat_trg_, trj_lat_trgsrc_, mcd=0)

                                tmp_batch_lat_dist_trgsrc1 = (tmp_batch_lat_dist_trgsrc1+tmp_batch_lat_dist_trgsrc2)/2
                                lat_dist_trgsrc1[0].append(tmp_batch_lat_dist_trgsrc1)
                                tmp_batch_lat_dist_trgsrc2 = (tmp_batch_lat_cdist_trgsrc1+tmp_batch_lat_cdist_trgsrc2)/2
                                lat_dist_trgsrc2[0].append(tmp_batch_lat_dist_trgsrc2)

                                batch_lat_dist_trgsrc1[0].append(tmp_batch_lat_dist_trgsrc1)
                                batch_lat_dist_trgsrc2[0].append(tmp_batch_lat_dist_trgsrc2)

                                batch_trg_spc_ = np.array(torch.index_select(batch_trg_[j,:,stdim:],0,spcidx_trg_[j,:flens_spc_trg_[j]]).cpu().data.numpy(), dtype=np.float64)
                                batch_trg_spc__ = np.array(torch.index_select(batch_trg_[j,:,stdim_:],0,spcidx_trg_[j,:flens_spc_trg_[j]]).cpu().data.numpy(), dtype=np.float64)

                                tmp_batch_mcdpow_trg_trg, _ = dtw.calc_mcd(batch_trg_spc_, np.array(torch.index_select(batch_trj_trg_trg_[i][j],0,spcidx_trg_[j,:flens_spc_trg_[j]]).cpu().data.numpy(), dtype=np.float64))
                                tmp_batch_mcd_trg_trg, _ = dtw.calc_mcd(batch_trg_spc__, np.array(torch.index_select(batch_trj_trg_trg_[i][j,:,1:],0,spcidx_trg_[j,:flens_spc_trg_[j]]).cpu().data.numpy(), dtype=np.float64))

                                tmp_batch_mcdpow_trg_src_trg, _ = dtw.calc_mcd(batch_trg_spc_, np.array(torch.index_select(batch_trj_trg_src_trg_[i][j],0,spcidx_trg_[j,:flens_spc_trg_[j]]).cpu().data.numpy(), dtype=np.float64))
                                tmp_batch_mcd_trg_src_trg, _ = dtw.calc_mcd(batch_trg_spc__, np.array(torch.index_select(batch_trj_trg_src_trg_[i][j,:,1:],0,spcidx_trg_[j,:flens_spc_trg_[j]]).cpu().data.numpy(), dtype=np.float64))

                                _, _, tmp_batch_mcdpow_trg_src, _ = dtw.dtw_org_to_trg(np.array(torch.index_select(batch_trj_trg_src_[i][j],0,spcidx_trg_[j,:flens_spc_trg_[j]]).cpu().data.numpy(), dtype=np.float64), np.array(torch.index_select(batch_trg_src_[j,:,stdim:],0,spcidx_trg_src_[j,:flens_spc_trg_src_[j]]).cpu().data.numpy(), dtype=np.float64))
                                _, _, tmp_batch_mcd_trg_src, _ = dtw.dtw_org_to_trg(np.array(torch.index_select(batch_trj_trg_src_[i][j,:,1:],0,spcidx_trg_[j,:flens_spc_trg_[j]]).cpu().data.numpy(), dtype=np.float64), np.array(torch.index_select(batch_trg_src_[j,:,stdim_:],0,spcidx_trg_src_[j,:flens_spc_trg_src_[j]]).cpu().data.numpy(), dtype=np.float64))

                                batch_src_spc_ = np.array(torch.index_select(batch_src_[j,:,stdim:],0,spcidx_src_[j,:flens_spc_src_[j]]).cpu().data.numpy(), dtype=np.float64)
                                batch_src_spc__ = np.array(torch.index_select(batch_src_[j,:,stdim_:],0,spcidx_src_[j,:flens_spc_src_[j]]).cpu().data.numpy(), dtype=np.float64)

                                tmp_batch_mcdpow_src_src, _ = dtw.calc_mcd(batch_src_spc_, np.array(torch.index_select(batch_trj_src_src_[i][j],0,spcidx_src_[j,:flens_spc_src_[j]]).cpu().data.numpy(), dtype=np.float64))
                                tmp_batch_mcd_src_src, _ = dtw.calc_mcd(batch_src_spc__, np.array(torch.index_select(batch_trj_src_src_[i][j,:,1:],0,spcidx_src_[j,:flens_spc_src_[j]]).cpu().data.numpy(), dtype=np.float64))

                                tmp_batch_mcdpow_src_trg_src, _ = dtw.calc_mcd(batch_src_spc_, np.array(torch.index_select(batch_trj_src_trg_src_[i][j],0,spcidx_src_[j,:flens_spc_src_[j]]).cpu().data.numpy(), dtype=np.float64))
                                tmp_batch_mcd_src_trg_src, _ = dtw.calc_mcd(batch_src_spc__, np.array(torch.index_select(batch_trj_src_trg_src_[i][j,:,1:],0,spcidx_src_[j,:flens_spc_src_[j]]).cpu().data.numpy(), dtype=np.float64))

                                _, _, tmp_batch_mcdpow_src_trg, _ = dtw.dtw_org_to_trg(np.array(torch.index_select(batch_trj_src_trg_[i][j],0,spcidx_src_[j,:flens_spc_src_[j]]).cpu().data.numpy(), dtype=np.float64), np.array(torch.index_select(batch_src_trg_[j,:,stdim:],0,spcidx_src_trg_[j,:flens_spc_src_trg_[j]]).cpu().data.numpy(), dtype=np.float64))
                                _, _, tmp_batch_mcd_src_trg, _ = dtw.dtw_org_to_trg(np.array(torch.index_select(batch_trj_src_trg_[i][j,:,1:],0,spcidx_src_[j,:flens_spc_src_[j]]).cpu().data.numpy(), dtype=np.float64), np.array(torch.index_select(batch_src_trg_[j,:,stdim_:],0,spcidx_src_trg_[j,:flens_spc_src_trg_[j]]).cpu().data.numpy(), dtype=np.float64))

                                batch_mcdpow_trg_trg[0].append(tmp_batch_mcdpow_trg_trg)
                                batch_mcd_trg_trg[0].append(tmp_batch_mcd_trg_trg)
                                batch_mcdpow_trg_src_trg[0].append(tmp_batch_mcdpow_trg_src_trg)
                                batch_mcd_trg_src_trg[0].append(tmp_batch_mcd_trg_src_trg)
                                batch_mcdpow_trg_src[0].append(tmp_batch_mcdpow_trg_src)
                                batch_mcd_trg_src[0].append(tmp_batch_mcd_trg_src)

                                batch_mcdpow_src_src[0].append(tmp_batch_mcdpow_src_src)
                                batch_mcd_src_src[0].append(tmp_batch_mcd_src_src)
                                batch_mcdpow_src_trg_src[0].append(tmp_batch_mcdpow_src_trg_src)
                                batch_mcd_src_trg_src[0].append(tmp_batch_mcd_src_trg_src)
                                batch_mcdpow_src_trg[0].append(tmp_batch_mcdpow_src_trg)
                                batch_mcd_src_trg[0].append(tmp_batch_mcd_src_trg)

                                mcdpow_trg_trg[i].append(tmp_batch_mcdpow_trg_trg)
                                mcd_trg_trg[i].append(tmp_batch_mcd_trg_trg)
                                mcdpow_trg_src_trg[i].append(tmp_batch_mcdpow_trg_src_trg)
                                mcd_trg_src_trg[i].append(tmp_batch_mcd_trg_src_trg)
                                mcdpow_trg_src[i].append(tmp_batch_mcdpow_trg_src)
                                mcd_trg_src[i].append(tmp_batch_mcd_trg_src)

                                mcdpow_src_src[i].append(tmp_batch_mcdpow_src_src)
                                mcd_src_src[i].append(tmp_batch_mcd_src_src)
                                mcdpow_src_trg_src[i].append(tmp_batch_mcdpow_src_trg_src)
                                mcd_src_trg_src[i].append(tmp_batch_mcd_src_trg_src)
                                mcdpow_src_trg[i].append(tmp_batch_mcdpow_src_trg)
                                mcd_src_trg[i].append(tmp_batch_mcd_src_trg)

                                logging.info("batch trgsrc loss %s %s = %.3f dB %.3f dB , %.3f dB %.3f dB , %.3f dB %.3f dB , %.3f %.3f" % (
                                        featfile_trg_[j], featfile_trg_src_[j], tmp_batch_mcdpow_trg_trg, tmp_batch_mcd_trg_trg, tmp_batch_mcdpow_trg_src_trg, tmp_batch_mcd_trg_src_trg,
                                            tmp_batch_mcdpow_trg_src, tmp_batch_mcd_trg_src, tmp_batch_lat_dist_trgsrc1, tmp_batch_lat_dist_trgsrc2))
                                logging.info("batch srctrg loss %s %s = %.3f dB %.3f dB , %.3f dB %.3f dB , %.3f dB %.3f dB , %.3f %.3f" % (
                                        featfile_src_[j], featfile_src_trg_[j], tmp_batch_mcdpow_src_src, tmp_batch_mcd_src_src, tmp_batch_mcdpow_src_trg_src, tmp_batch_mcd_src_trg_src,
                                            tmp_batch_mcdpow_src_trg, tmp_batch_mcd_src_trg, tmp_batch_lat_dist_srctrg1, tmp_batch_lat_dist_srctrg2))

                            batch_mcdpow_src_src[i] = np.mean(batch_mcdpow_src_src[i])
                            batch_mcd_src_src[i] = np.mean(batch_mcd_src_src[i])
                            batch_mcdpow_src_trg_src[i] = np.mean(batch_mcdpow_src_trg_src[i])
                            batch_mcd_src_trg_src[i] = np.mean(batch_mcd_src_trg_src[i])
                            batch_mcdpow_src_trg[i] = np.mean(batch_mcdpow_src_trg[i])
                            batch_mcd_src_trg[i] = np.mean(batch_mcd_src_trg[i])
                            batch_mcdpow_trg_trg[i] = np.mean(batch_mcdpow_trg_trg[i])
                            batch_mcd_trg_trg[i] = np.mean(batch_mcd_trg_trg[i])
                            batch_mcdpow_trg_src_trg[i] = np.mean(batch_mcdpow_trg_src_trg[i])
                            batch_mcd_trg_src_trg[i] = np.mean(batch_mcd_trg_src_trg[i])
                            batch_mcdpow_trg_src[i] = np.mean(batch_mcdpow_trg_src[i])
                            batch_mcd_trg_src[i] = np.mean(batch_mcd_trg_src[i])
                            batch_lat_dist_srctrg1[i] = np.mean(batch_lat_dist_srctrg1[i])
                            batch_lat_dist_srctrg2[i] = np.mean(batch_lat_dist_srctrg2[i])
                            batch_lat_dist_trgsrc1[i] = np.mean(batch_lat_dist_trgsrc1[i])
                            batch_lat_dist_trgsrc2[i] = np.mean(batch_lat_dist_trgsrc2[i])

                        for j in range(n_batch_utt_):
                            _, tmp_batch_loss_mcd_trg_trg, _ = criterion_mcd(batch_trj_trg_trg_[i][j,:flens_trg_[j]], batch_trg_[j,:flens_trg_[j],stdim:], L2=False, GV=False)
                            _, tmp_batch_loss_mcd_trg_src, _ = criterion_mcd(batch_trj_trg_src_[i][j,:flens_trg_[j]], batch_trg_[j,:flens_trg_[j],stdim:], L2=False, GV=False)

                            _, tmp_batch_loss_mcd_src_src, _ = criterion_mcd(batch_trj_src_src_[i][j,:flens_src_[j]], batch_src_[j,:flens_src_[j],stdim:], L2=False, GV=False)
                            _, tmp_batch_loss_mcd_src_trg, _ = criterion_mcd(batch_trj_src_trg_[i][j,:flens_src_[j]], batch_src_[j,:flens_src_[j],stdim:], L2=False, GV=False)

                            _, tmp_batch_loss_mcd_trg_src_trg, _ = criterion_mcd(batch_trj_trg_src_trg_[i][j,:flens_trg_[j]], batch_trg_[j,:flens_trg_[j],stdim:], L2=False, GV=False)
                            _, tmp_batch_loss_mcd_src_trg_src, _ = criterion_mcd(batch_trj_src_trg_src_[i][j,:flens_src_[j]], batch_src_[j,:flens_src_[j],stdim:], L2=False, GV=False)

                            tmp_batch_loss_lat_trg = loss_vae(batch_lat_trg_[i][j,:flens_trg_[j]], lat_dim=args.lat_dim)
                            tmp_batch_loss_lat_src = loss_vae(batch_lat_src_[i][j,:flens_src_[j]], lat_dim=args.lat_dim)

                            tmp_batch_loss_lat_trg_cv = loss_vae(batch_lat_trg_src_[i][j,:flens_trg_[j]], lat_dim=args.lat_dim)
                            tmp_batch_loss_lat_src_cv = loss_vae(batch_lat_src_trg_[i][j,:flens_src_[j]], lat_dim=args.lat_dim)

                            if j > 0:
                                batch_loss_mcd_trg_trg[i] = torch.cat((batch_loss_mcd_trg_trg[i], tmp_batch_loss_mcd_trg_trg.unsqueeze(0)))
                                batch_loss_mcd_trg_src[i] = torch.cat((batch_loss_mcd_trg_src[i], tmp_batch_loss_mcd_trg_src.unsqueeze(0)))

                                batch_loss_mcd_src_src[i] = torch.cat((batch_loss_mcd_src_src[i], tmp_batch_loss_mcd_src_src.unsqueeze(0)))
                                batch_loss_mcd_src_trg[i] = torch.cat((batch_loss_mcd_src_trg[i], tmp_batch_loss_mcd_src_trg.unsqueeze(0)))

                                batch_loss_mcd_trg_src_trg[i] = torch.cat((batch_loss_mcd_trg_src_trg[i], tmp_batch_loss_mcd_trg_src_trg.unsqueeze(0)))
                                batch_loss_mcd_src_trg_src[i] = torch.cat((batch_loss_mcd_src_trg_src[i], tmp_batch_loss_mcd_src_trg_src.unsqueeze(0)))

                                batch_loss_lat_trg[i] = torch.cat((batch_loss_lat_trg[i], tmp_batch_loss_lat_trg.unsqueeze(0)))
                                batch_loss_lat_src[i] = torch.cat((batch_loss_lat_src[i], tmp_batch_loss_lat_src.unsqueeze(0)))

                                batch_loss_lat_trg_cv[i] = torch.cat((batch_loss_lat_trg_cv[i], tmp_batch_loss_lat_trg_cv.unsqueeze(0)))
                                batch_loss_lat_src_cv[i] = torch.cat((batch_loss_lat_src_cv[i], tmp_batch_loss_lat_src_cv.unsqueeze(0)))
                            else:
                                batch_loss_mcd_trg_trg[i] = tmp_batch_loss_mcd_trg_trg.unsqueeze(0)
                                batch_loss_mcd_trg_src[i] = tmp_batch_loss_mcd_trg_src.unsqueeze(0)

                                batch_loss_mcd_src_src[i] = tmp_batch_loss_mcd_src_src.unsqueeze(0)
                                batch_loss_mcd_src_trg[i] = tmp_batch_loss_mcd_src_trg.unsqueeze(0)

                                batch_loss_mcd_trg_src_trg[i] = tmp_batch_loss_mcd_trg_src_trg.unsqueeze(0)
                                batch_loss_mcd_src_trg_src[i] = tmp_batch_loss_mcd_src_trg_src.unsqueeze(0)

                                batch_loss_lat_trg[i] = tmp_batch_loss_lat_trg.unsqueeze(0)
                                batch_loss_lat_src[i] = tmp_batch_loss_lat_src.unsqueeze(0)

                                batch_loss_lat_trg_cv[i] = tmp_batch_loss_lat_trg_cv.unsqueeze(0)
                                batch_loss_lat_src_cv[i] = tmp_batch_loss_lat_src_cv.unsqueeze(0)

                        batch_loss_mcd_trg_trg[i] = torch.mean(batch_loss_mcd_trg_trg[i])
                        batch_loss_mcd_trg_src_trg[i] = torch.mean(batch_loss_mcd_trg_src_trg[i])
                        batch_loss_mcd_trg_src[i] = torch.mean(batch_loss_mcd_trg_src[i])
                        batch_loss_lat_trg[i] = torch.mean(batch_loss_lat_trg[i])
                        batch_loss_lat_trg_cv[i] = torch.mean(batch_loss_lat_trg_cv[i])

                        batch_loss_mcd_src_src[i] = torch.mean(batch_loss_mcd_src_src[i])
                        batch_loss_mcd_src_trg_src[i] = torch.mean(batch_loss_mcd_src_trg_src[i])
                        batch_loss_mcd_src_trg[i] = torch.mean(batch_loss_mcd_src_trg[i])
                        batch_loss_lat_src[i] = torch.mean(batch_loss_lat_src[i])
                        batch_loss_lat_src_cv[i] = torch.mean(batch_loss_lat_src_cv[i])

                        loss_mcd_trg_trg[i].append(batch_loss_mcd_trg_trg[i].item())
                        loss_mcd_trg_src[i].append(batch_loss_mcd_trg_src[i].item())

                        loss_mcd_src_src[i].append(batch_loss_mcd_src_src[i].item())
                        loss_mcd_src_trg[i].append(batch_loss_mcd_src_trg[i].item())

                        loss_mcd_trg_src_trg[i].append(batch_loss_mcd_trg_src_trg[i].item())
                        loss_mcd_src_trg_src[i].append(batch_loss_mcd_src_trg_src[i].item())

                        loss_lat_trg[i].append(batch_loss_lat_trg[i].item())
                        loss_lat_src[i].append(batch_loss_lat_src[i].item())

                        loss_lat_trg_cv[i].append(batch_loss_lat_trg_cv[i].item())
                        loss_lat_src_cv[i].append(batch_loss_lat_src_cv[i].item())

                        if i > 0: 
                            if not half_cyc:
                                batch_loss += batch_loss_mcd_trg_trg[i] + batch_loss_mcd_src_src[i] + batch_loss_mcd_trg_src_trg[i] + batch_loss_mcd_src_trg_src[i] + batch_loss_lat_trg[i] + batch_loss_lat_src[i] + batch_loss_lat_trg_cv[i] + batch_loss_lat_src_cv[i]
                            else:
                                batch_loss += batch_loss_mcd_trg_trg[i] + batch_loss_mcd_src_src[i] + batch_loss_lat_trg[i] + batch_loss_lat_src[i]
                        else:
                            if not half_cyc:
                                batch_loss = batch_loss_mcd_trg_trg[0] + batch_loss_mcd_src_src[0] + batch_loss_mcd_trg_src_trg[0] + batch_loss_mcd_src_trg_src[0] + batch_loss_lat_trg[0] + batch_loss_lat_src[0] + batch_loss_lat_trg_cv[0] + batch_loss_lat_src_cv[0]
                            else:
                                batch_loss = batch_loss_mcd_trg_trg[0] + batch_loss_mcd_src_src[0] + batch_loss_lat_trg[0] + batch_loss_lat_src[0]

                    loss.append(batch_loss.item())
                    text_log = "%.3f ;; " % batch_loss.item()
                    for i in range(n_ev_cyc):
                        text_log += "[%d] %.3f %.3f %.3f %.3f %.3f %.3f ; %.3f %.3f %.3f %.3f ; %.3f dB %.3f dB , %.3f dB %.3f dB , %.3f dB %.3f dB , %.3f %.3f ; %.3f dB %.3f dB , %.3f dB %.3f dB , %.3f dB %.3f dB , %.3f %.3f ;; " % (
                                     i+1, batch_loss_mcd_trg_trg[i].item(), batch_loss_mcd_trg_src_trg[i].item(), batch_loss_mcd_trg_src[i].item(),
                                         batch_loss_mcd_src_src[i].item(), batch_loss_mcd_src_trg_src[i].item(), batch_loss_mcd_src_trg[i].item(),
                                             batch_loss_lat_trg[i].item(), batch_loss_lat_trg_cv[i].item(), batch_loss_lat_src[i].item(), batch_loss_lat_src_cv[i].item(),
                                                 batch_mcdpow_trg_trg[i], batch_mcd_trg_trg[i], batch_mcdpow_trg_src_trg[i], batch_mcd_trg_src_trg[i],
                                                         batch_mcdpow_trg_src[i], batch_mcd_trg_src[i], batch_lat_dist_trgsrc1[i], batch_lat_dist_trgsrc2[i], batch_mcdpow_src_src[i], batch_mcd_src_src[i],
                                                                 batch_mcdpow_src_trg_src[i], batch_mcd_src_trg_src[i], batch_mcdpow_src_trg[i], batch_mcd_src_trg[i], batch_lat_dist_srctrg1[i], batch_lat_dist_srctrg2[i])
                    logging.info("batch eval loss [%d] = %s  (%.3f sec)" % (c_idx+1, text_log, time.time() - start))
                    total.append(time.time() - start)
                eval_loss = np.mean(loss)
            for i in range(n_ev_cyc):
                eval_loss_mcd_trg_trg[i] = np.mean(loss_mcd_trg_trg[i])
                eval_loss_mcd_trg_src_trg[i] = np.mean(loss_mcd_trg_src_trg[i])
                eval_loss_mcd_trg_src[i] = np.mean(loss_mcd_trg_src[i])
                eval_loss_mcd_src_src[i] = np.mean(loss_mcd_src_src[i])
                eval_loss_mcd_src_trg_src[i] = np.mean(loss_mcd_src_trg_src[i])
                eval_loss_mcd_src_trg[i] = np.mean(loss_mcd_src_trg[i])
                eval_loss_lat_src_cv[i] = np.mean(loss_lat_src_cv[i])
                eval_loss_lat_trg_cv[i] = np.mean(loss_lat_trg_cv[i])
                eval_loss_lat_src[i] = np.mean(loss_lat_src[i])
                eval_loss_lat_trg[i] = np.mean(loss_lat_trg[i])
                eval_gv_trg_trg[i] = np.mean(np.sqrt(np.square(np.log(np.mean(gv_trg_trg[i], axis=0))-np.log(gv_trg_mean))))
                eval_mcdpow_trg_trg[i] = np.mean(mcdpow_trg_trg[i])
                eval_mcd_trg_trg[i] = np.mean(mcd_trg_trg[i])
                eval_gv_trg_src_trg[i] = np.mean(np.sqrt(np.square(np.log(np.mean(gv_trg_src_trg[i], axis=0))-np.log(gv_trg_mean))))
                eval_mcdpow_trg_src_trg[i] = np.mean(mcdpow_trg_src_trg[i])
                eval_mcd_trg_src_trg[i] = np.mean(mcd_trg_src_trg[i])
                eval_gv_trg_src[i] = np.mean(np.sqrt(np.square(np.log(np.mean(gv_trg_src[i], axis=0))-np.log(gv_src_mean))))
                eval_mcdpow_trg_src[i] = np.mean(mcdpow_trg_src[i])
                eval_mcdpowstd_trg_src[i] = np.std(mcdpow_trg_src[i])
                eval_mcd_trg_src[i] = np.mean(mcd_trg_src[i])
                eval_mcdstd_trg_src[i] = np.std(mcd_trg_src[i])
                eval_lat_dist_trgsrc1[i] = np.mean(lat_dist_trgsrc1[i])
                eval_lat_dist_trgsrc2[i] = np.mean(lat_dist_trgsrc2[i])
                eval_gv_src_src[i] = np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_src[i], axis=0))-np.log(gv_src_mean))))
                eval_mcdpow_src_src[i] = np.mean(mcdpow_src_src[i])
                eval_mcd_src_src[i] = np.mean(mcd_src_src[i])
                eval_gv_src_trg_src[i] = np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_trg_src[i], axis=0))-np.log(gv_src_mean))))
                eval_mcdpow_src_trg_src[i] = np.mean(mcdpow_src_trg_src[i])
                eval_mcd_src_trg_src[i] = np.mean(mcd_src_trg_src[i])
                eval_gv_src_trg[i] = np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_trg[i], axis=0))-np.log(gv_trg_mean))))
                eval_mcdpow_src_trg[i] = np.mean(mcdpow_src_trg[i])
                eval_mcdpowstd_src_trg[i] = np.std(mcdpow_src_trg[i])
                eval_mcd_src_trg[i] = np.mean(mcd_src_trg[i])
                eval_mcdstd_src_trg[i] = np.std(mcd_src_trg[i])
                eval_lat_dist_srctrg1[i] = np.mean(lat_dist_srctrg1[i])
                eval_lat_dist_srctrg2[i] = np.mean(lat_dist_srctrg2[i])
            text_log = "%.3f ;; " % eval_loss
            for i in range(n_ev_cyc):
                text_log += "[%d] %.3f %.3f %.3f %.3f %.3f %.3f ; %.3f %.3f %.3f %.3f ; %.6f %.3f dB %.6f dB , %.3f %.3f dB %.3f dB , %.6f %.3f dB (+- %.3f) %.6f dB (+- %.3f) , %.6f %.6f ; %.6f %.3f dB %.6f dB , %.3f %.3f dB %.3f dB , %.6f %.3f dB (+- %.3f) %.6f dB (+- %.3f) , %.6f %.6f ;; " % (
                             i+1, eval_loss_mcd_trg_trg[i], eval_loss_mcd_trg_src_trg[i], eval_loss_mcd_trg_src[i],
                                 eval_loss_mcd_src_src[i], eval_loss_mcd_src_trg_src[i], eval_loss_mcd_src_trg[i],
                                     eval_loss_lat_trg[i], eval_loss_lat_trg_cv[i], eval_loss_lat_src[i], eval_loss_lat_src_cv[i],
                                         eval_gv_trg_trg[i], eval_mcdpow_trg_trg[i], eval_mcd_trg_trg[i],
                                             eval_gv_trg_src_trg[i], eval_mcdpow_trg_src_trg[i], eval_mcd_trg_src_trg[i],
                                                 eval_gv_trg_src[i], eval_mcdpow_trg_src[i], eval_mcdpowstd_trg_src[i], eval_mcd_trg_src[i], eval_mcdstd_trg_src[i],
                                                     eval_lat_dist_trgsrc1[i], eval_lat_dist_trgsrc2[i], eval_gv_src_src[i], eval_mcdpow_src_src[i], eval_mcd_src_src[i],
                                                         eval_gv_src_trg_src[i], eval_mcdpow_src_trg_src[i], eval_mcd_src_trg_src[i],
                                                             eval_gv_src_trg[i], eval_mcdpow_src_trg[i], eval_mcdpowstd_src_trg[i], eval_mcd_src_trg[i], eval_mcdstd_src_trg[i], eval_lat_dist_srctrg1[i], eval_lat_dist_srctrg2[i])
            logging.info("(EPOCH:%d) average evaluation loss = %s  (%.3f min., %.3f sec / batch)" % (epoch_idx + 1, text_log, np.sum(total) / 60.0, np.mean(total)))
            if (eval_mcdpow_src_trg[0]+eval_mcdpowstd_src_trg[0]+eval_mcd_src_trg[0]+eval_mcdstd_src_trg[0]) <= (min_eval_mcdpow_src_trg+min_eval_mcdpowstd_src_trg+min_eval_mcd_src_trg+min_eval_mcdstd_src_trg):
                min_eval_loss_mcd_trg_trg = eval_loss_mcd_trg_trg[0]
                min_eval_loss_mcd_trg_src_trg = eval_loss_mcd_trg_src_trg[0]
                min_eval_loss_mcd_trg_src = eval_loss_mcd_trg_src[0]
                min_eval_loss_mcd_src_src = eval_loss_mcd_src_src[0]
                min_eval_loss_mcd_src_trg_src = eval_loss_mcd_src_trg_src[0]
                min_eval_loss_mcd_src_trg = eval_loss_mcd_src_trg[0]
                min_eval_loss_lat_src = eval_loss_lat_src[0]
                min_eval_loss_lat_trg = eval_loss_lat_trg[0]
                min_eval_loss_lat_src_cv = eval_loss_lat_src_cv[0]
                min_eval_loss_lat_trg_cv = eval_loss_lat_trg_cv[0]
                min_eval_gv_trg_trg = eval_gv_trg_trg[0]
                min_eval_mcdpow_trg_trg = eval_mcdpow_trg_trg[0]
                min_eval_mcd_trg_trg = eval_mcd_trg_trg[0]
                min_eval_gv_trg_src_trg = eval_gv_trg_src_trg[0]
                min_eval_mcdpow_trg_src_trg = eval_mcdpow_trg_src_trg[0]
                min_eval_mcd_trg_src_trg = eval_mcd_trg_src_trg[0]
                min_eval_gv_trg_src = eval_gv_trg_src[0]
                min_eval_mcdpow_trg_src = eval_mcdpow_trg_src[0]
                min_eval_mcdpowstd_trg_src = eval_mcdpowstd_trg_src[0]
                min_eval_mcd_trg_src = eval_mcd_trg_src[0]
                min_eval_mcdstd_trg_src = eval_mcdstd_trg_src[0]
                min_eval_lat_dist_trgsrc1 = eval_lat_dist_trgsrc1[0]
                min_eval_lat_dist_trgsrc2 = eval_lat_dist_trgsrc2[0]
                min_eval_gv_src_src = eval_gv_src_src[0]
                min_eval_mcdpow_src_src = eval_mcdpow_src_src[0]
                min_eval_mcd_src_src = eval_mcd_src_src[0]
                min_eval_gv_src_trg_src = eval_gv_src_trg_src[0]
                min_eval_mcdpow_src_trg_src = eval_mcdpow_src_trg_src[0]
                min_eval_mcd_src_trg_src = eval_mcd_src_trg_src[0]
                min_eval_gv_src_trg = eval_gv_src_trg[0]
                min_eval_mcdpow_src_trg = eval_mcdpow_src_trg[0]
                min_eval_mcdpowstd_src_trg = eval_mcdpowstd_src_trg[0]
                min_eval_mcd_src_trg = eval_mcd_src_trg[0]
                min_eval_mcdstd_src_trg = eval_mcdstd_src_trg[0]
                min_eval_lat_dist_srctrg1 = eval_lat_dist_srctrg1[0]
                min_eval_lat_dist_srctrg2 = eval_lat_dist_srctrg2[0]
                min_idx = epoch_idx
            text_log = "%.3f %.3f %.3f %.3f %.3f %.3f ; %.3f %.3f %.3f %.3f ; %.6f %.3f dB %.6f dB , %.3f %.3f dB %.3f dB , %.6f %.3f dB (+- %.3f) %.6f dB (+- %.3f) , %.6f %.6f ; %.6f %.3f dB %.6f dB , %.3f %.3f dB %.3f dB , %.6f %.3f dB (+- %.3f) %.6f dB (+- %.3f) , %.6f %.6f ;; " % (
                         min_eval_loss_mcd_trg_trg, min_eval_loss_mcd_trg_src_trg, min_eval_loss_mcd_trg_src,
                             min_eval_loss_mcd_src_src, min_eval_loss_mcd_src_trg_src, min_eval_loss_mcd_src_trg,
                                 min_eval_loss_lat_trg, min_eval_loss_lat_trg_cv, min_eval_loss_lat_src, min_eval_loss_lat_src_cv,
                                     min_eval_gv_trg_trg, min_eval_mcdpow_trg_trg, min_eval_mcd_trg_trg,
                                         min_eval_gv_trg_src_trg, min_eval_mcdpow_trg_src_trg, min_eval_mcd_trg_src_trg,
                                             min_eval_gv_trg_src, min_eval_mcdpow_trg_src, min_eval_mcdpowstd_trg_src, min_eval_mcd_trg_src, min_eval_mcdstd_trg_src,
                                                 min_eval_lat_dist_trgsrc1, min_eval_lat_dist_trgsrc2, min_eval_gv_src_src, min_eval_mcdpow_src_src, min_eval_mcd_src_src,
                                                     min_eval_gv_src_trg_src, min_eval_mcdpow_src_trg_src, min_eval_mcd_src_trg_src,
                                                         min_eval_gv_src_trg, min_eval_mcdpow_src_trg, min_eval_mcdpowstd_src_trg, min_eval_mcd_src_trg, min_eval_mcdstd_src_trg, min_eval_lat_dist_srctrg1, min_eval_lat_dist_srctrg2)
            logging.info("min_eval_acc= %s min_idx=%d" % (text_log, min_idx+1))
            loss = []
            loss_mcd_trg_trg = []
            loss_mcd_trg_src_trg = []
            loss_mcd_trg_src = []
            loss_mcd_src_src = []
            loss_mcd_src_trg_src = []
            loss_mcd_src_trg = []
            loss_lat_src = []
            loss_lat_trg = []
            loss_lat_src_cv = []
            loss_lat_trg_cv = []
            gv_trg_trg = []
            mcdpow_trg_trg = []
            mcd_trg_trg = []
            gv_trg_src_trg = []
            mcdpow_trg_src_trg = []
            mcd_trg_src_trg = []
            gv_trg_src = []
            mcdpow_trg_src = []
            mcd_trg_src = []
            lat_dist_trgsrc1 = []
            lat_dist_trgsrc2 = []
            gv_src_src = []
            mcdpow_src_src = []
            mcd_src_src = []
            gv_src_trg_src = []
            mcdpow_src_trg_src = []
            mcd_src_trg_src = []
            gv_src_trg = []
            mcdpow_src_trg = []
            mcd_src_trg = []
            lat_dist_srctrg1 = []
            lat_dist_srctrg2 = []
            for i in range(args.n_cyc):
                loss_mcd_trg_trg.append([])
                loss_mcd_trg_src_trg.append([])
                loss_mcd_trg_src.append([])
                loss_mcd_src_src.append([])
                loss_mcd_src_trg_src.append([])
                loss_mcd_src_trg.append([])
                loss_lat_src.append([])
                loss_lat_trg.append([])
                loss_lat_src_cv.append([])
                loss_lat_trg_cv.append([])
                gv_trg_trg.append([])
                mcdpow_trg_trg.append([])
                mcd_trg_trg.append([])
                gv_trg_src_trg.append([])
                mcdpow_trg_src_trg.append([])
                mcd_trg_src_trg.append([])
                gv_trg_src.append([])
                mcdpow_trg_src.append([])
                mcd_trg_src.append([])
                lat_dist_trgsrc1.append([])
                lat_dist_trgsrc2.append([])
                gv_src_src.append([])
                mcdpow_src_src.append([])
                mcd_src_src.append([])
                gv_src_trg_src.append([])
                mcdpow_src_trg_src.append([])
                mcd_src_trg_src.append([])
                gv_src_trg.append([])
                mcdpow_src_trg.append([])
                mcd_src_trg.append([])
                lat_dist_srctrg1.append([])
                lat_dist_srctrg2.append([])
            total = []
            iter_count = 0 
            epoch_idx += 1
            np.random.set_state(numpy_random_state)
            torch.set_rng_state(torch_random_state)
            model_encoder.train()
            model_decoder.train()
            for param in model_encoder.parameters():
                param.requires_grad = True
            for param in model_decoder.parameters():
                param.requires_grad = True
            for param in model_encoder.scale_in.parameters():
                param.requires_grad = False
            for param in model_decoder.scale_out.parameters():
                param.requires_grad = False
            # start next epoch
            if epoch_idx < args.epoch_count:
                start = time.time()
                logging.info("==%d EPOCH==" % (epoch_idx+1))
                logging.info("Training data")
        # feedforward and backpropagate current batch
        if epoch_idx < args.epoch_count:
            logging.info("%d iteration [%d]" % (iter_idx+1, epoch_idx+1))
            
            if args.batch_size > 0: # frame-length mini-batch
                for i in range(n_batch_utt):
                    logging.info("%s %s %d %d %d %d %d %d %d %d %d %d" % (
                        featfile_src[i], featfile_src_trg[i], flens_src[i], flens_src_trg[i], flens_spc_src[i], flens_spc_src_trg[i],
                            src_idx_s, src_idx_e, spcidx_src_s_idx[i], spcidx_src_e_idx[i], spcidx_src[i,spcidx_src_s_idx[i]].item(), spcidx_src[i,spcidx_src_e_idx[i]].item()))

                if src_idx_s > 0 and prev_featfile_src == featfile_src and iter_count > 0:
                    for i in range(args.n_cyc):
                        if i > 0:
                            batch_lat_src[i], y_in_pp_src[i], h_in_pp_src[i] = model_encoder(torch.cat((batch_src[:,src_idx_s:src_idx_e+1,:stdim], batch_trj_src_trg_src[i-1]),2), Variable(y_in_pp_src[i].data).detach(), h_in=Variable(h_in_pp_src[i].data).detach(), do=True, clamp_vae=True, lat_dim=args.lat_dim)
                            batch_trj_src_src[i], y_in_src_src[i], h_in_src_src[i] = model_decoder(torch.cat((batch_src_src_code, sampling_vae_batch(batch_lat_src[i], lat_dim=args.lat_dim, training=True)),2), Variable(y_in_src_src[i].data).detach(), h_in=Variable(h_in_src_src[i].data).detach(), do=True)
                            batch_trj_src_trg[i], y_in_src_trg[i], h_in_src_trg[i] = model_decoder(torch.cat((batch_src_trg_code, sampling_vae_batch(batch_lat_src[i], lat_dim=args.lat_dim, training=True)),2), Variable(y_in_src_trg[i].data).detach(), h_in=Variable(h_in_src_trg[i].data).detach(), do=True)
                            batch_lat_src_trg[i], y_in_pp_src_trg[i], h_in_pp_src_trg[i] = model_encoder(torch.cat((batch_cv_src[:,src_idx_s:src_idx_e+1], batch_trj_src_trg[i]),2), Variable(y_in_pp_src_trg[i].data).detach(), h_in=Variable(h_in_pp_src_trg[i].data).detach(), do=True, clamp_vae=True, lat_dim=args.lat_dim)
                            batch_trj_src_trg_src[i], y_in_src_trg_src[i], h_in_src_trg_src[i] = model_decoder(torch.cat((batch_src_src_code, sampling_vae_batch(batch_lat_src_trg[i], lat_dim=args.lat_dim, training=True)),2), Variable(y_in_src_trg_src[i].data).detach(), h_in=Variable(h_in_src_trg_src[i].data).detach(), do=True)
                        else:
                            batch_lat_src[0], y_in_pp_src[0], h_in_pp_src[0] = model_encoder(batch_src[:,src_idx_s:src_idx_e+1], Variable(y_in_pp_src[0].data).detach(), h_in=Variable(h_in_pp_src[0].data).detach(), do=True, clamp_vae=True, lat_dim=args.lat_dim)
                            batch_trj_src_src[0], y_in_src_src[0], h_in_src_src[0] = model_decoder(torch.cat((batch_src_src_code, sampling_vae_batch(batch_lat_src[0], lat_dim=args.lat_dim, training=True)),2), Variable(y_in_src_src[0].data).detach(), h_in=Variable(h_in_src_src[0].data).detach(), do=True)
                            batch_trj_src_trg[0], y_in_src_trg[0], h_in_src_trg[0] = model_decoder(torch.cat((batch_src_trg_code, sampling_vae_batch(batch_lat_src[0], lat_dim=args.lat_dim, training=True)),2), Variable(y_in_src_trg[0].data).detach(), h_in=Variable(h_in_src_trg[0].data).detach(), do=True)
                            batch_lat_src_trg[0], y_in_pp_src_trg[0], h_in_pp_src_trg[0] = model_encoder(torch.cat((batch_cv_src[:,src_idx_s:src_idx_e+1], batch_trj_src_trg[0]),2), Variable(y_in_pp_src_trg[0].data).detach(), h_in=Variable(h_in_pp_src_trg[0].data).detach(), do=True, clamp_vae=True, lat_dim=args.lat_dim)
                            batch_trj_src_trg_src[0], y_in_src_trg_src[0], h_in_src_trg_src[0] = model_decoder(torch.cat((batch_src_src_code, sampling_vae_batch(batch_lat_src_trg[0], lat_dim=args.lat_dim, training=True)),2), Variable(y_in_src_trg_src[0].data).detach(), h_in=Variable(h_in_src_trg_src[0].data).detach(), do=True)
                            tmp_src_src = np.concatenate((tmp_src_src, batch_trj_src_src[0][:,:,1:].cpu().data.numpy()), axis=1)
                            tmp_src_trg = np.concatenate((tmp_src_trg, batch_trj_src_trg[0][:,:,1:].cpu().data.numpy()), axis=1)
                            tmp_src_trg_src = np.concatenate((tmp_src_trg_src, batch_trj_src_trg_src[0][:,:,1:].cpu().data.numpy()), axis=1)
                            trj_src_trg = torch.cat((trj_src_trg, batch_trj_src_trg[0]),1)
                            trj_lat_src = torch.cat((trj_lat_src, batch_lat_src[0]),1)
                else:
                    if n_batch_utt == args.batch_size_utt:
                        y_in_pp_ = y_in_pp
                        y_in_src_ = y_in_src
                        y_in_trg_ = y_in_trg
                    else:
                        y_in_pp_ = y_in_pp_mod
                        y_in_src_ = y_in_src_mod
                        y_in_trg_ = y_in_trg_mod
                    for i in range(args.n_cyc):
                        if i > 0:
                            batch_lat_src[i], y_in_pp_src[i], h_in_pp_src[i] = model_encoder(torch.cat((batch_src[:,src_idx_s:src_idx_e+1,:stdim], batch_trj_src_trg_src[i-1]),2), y_in_pp_, do=True, clamp_vae=True, lat_dim=args.lat_dim)
                            batch_trj_src_src[i], y_in_src_src[i], h_in_src_src[i] = model_decoder(torch.cat((batch_src_src_code, sampling_vae_batch(batch_lat_src[i], lat_dim=args.lat_dim, training=True)),2), y_in_src_, do=True)
                            batch_trj_src_trg[i], y_in_src_trg[i], h_in_src_trg[i] = model_decoder(torch.cat((batch_src_trg_code, sampling_vae_batch(batch_lat_src[i], lat_dim=args.lat_dim, training=True)),2), y_in_trg_, do=True)
                            batch_lat_src_trg[i], y_in_pp_src_trg[i], h_in_pp_src_trg[i] = model_encoder(torch.cat((batch_cv_src[:,src_idx_s:src_idx_e+1], batch_trj_src_trg[i]),2), y_in_pp_, do=True, clamp_vae=True, lat_dim=args.lat_dim)
                            batch_trj_src_trg_src[i], y_in_src_trg_src[i], h_in_src_trg_src[i] = model_decoder(torch.cat((batch_src_src_code, sampling_vae_batch(batch_lat_src_trg[i], lat_dim=args.lat_dim, training=True)),2), y_in_src_, do=True)
                        else:
                            batch_lat_src[0], y_in_pp_src[0], h_in_pp_src[0] = model_encoder(batch_src[:,src_idx_s:src_idx_e+1], y_in_pp_, do=True, clamp_vae=True, lat_dim=args.lat_dim)
                            batch_trj_src_src[0], y_in_src_src[0], h_in_src_src[0] = model_decoder(torch.cat((batch_src_src_code, sampling_vae_batch(batch_lat_src[0], lat_dim=args.lat_dim, training=True)),2), y_in_src_, do=True)
                            batch_trj_src_trg[0], y_in_src_trg[0], h_in_src_trg[0] = model_decoder(torch.cat((batch_src_trg_code, sampling_vae_batch(batch_lat_src[0], lat_dim=args.lat_dim, training=True)),2), y_in_trg_, do=True)
                            batch_lat_src_trg[0], y_in_pp_src_trg[0], h_in_pp_src_trg[0] = model_encoder(torch.cat((batch_cv_src[:,src_idx_s:src_idx_e+1], batch_trj_src_trg[0]),2), y_in_pp_, do=True, clamp_vae=True, lat_dim=args.lat_dim)
                            batch_trj_src_trg_src[0], y_in_src_trg_src[0], h_in_src_trg_src[0] = model_decoder(torch.cat((batch_src_src_code, sampling_vae_batch(batch_lat_src_trg[0], lat_dim=args.lat_dim, training=True)),2), y_in_src_, do=True)
                            if iter_count > 0:
                                for j in range(n_batch_utt):
                                    if os.path.basename(os.path.dirname(prev_featfile_src[j])) == args.spk_src:
                                        gv_src_src[i].append(np.var(tmp_src_src[j,:prev_flens_src[j]], axis=0))
                                        gv_src_trg[i].append(np.var(tmp_src_trg[j,:prev_flens_src[j]], axis=0))
                                        gv_src_trg_src[i].append(np.var(tmp_src_trg_src[j,:prev_flens_src[j]], axis=0))
                                    else:
                                        gv_trg_trg[i].append(np.var(tmp_src_src[j,:prev_flens_src[j]], axis=0))
                                        gv_trg_src[i].append(np.var(tmp_src_trg[j,:prev_flens_src[j]], axis=0))
                                        gv_trg_src_trg[i].append(np.var(tmp_src_trg_src[j,:prev_flens_src[j]], axis=0))
                            tmp_src_src = batch_trj_src_src[0][:,:,1:].cpu().data.numpy()
                            tmp_src_trg = batch_trj_src_trg[0][:,:,1:].cpu().data.numpy()
                            tmp_src_trg_src = batch_trj_src_trg_src[0][:,:,1:].cpu().data.numpy()
                            trj_src_trg = batch_trj_src_trg[0]
                            trj_lat_src = batch_lat_src[0]
                prev_featfile_src = featfile_src

                if len(select_utt_idx) > 0:
                    for i in range(args.n_cyc):
                        batch_mcdpow_src_src[i] = []
                        batch_mcd_src_src[i] = []
                        batch_mcdpow_src_trg_src[i] = []
                        batch_mcd_src_trg_src[i] = []

                    for i in range(args.n_cyc):
                        for k, j in enumerate(select_utt_idx):
                            src_idx_e_ = src_idx_s + flen_acc[j]
                            _, tmp_batch_loss_mcd_src_src, _ = criterion_mcd(batch_trj_src_src[i][j,:flen_acc[j]], batch_src[j,src_idx_s:src_idx_e_,stdim:], L2=False, GV=False)
                            _, tmp_batch_loss_mcd_src_trg_src, _ = criterion_mcd(batch_trj_src_trg_src[i][j,:flen_acc[j]], batch_src[j,src_idx_s:src_idx_e_,stdim:], L2=False, GV=False)
                            _, tmp_batch_loss_mcd_src_trg, _ = criterion_mcd(batch_trj_src_trg[i][j,:flen_acc[j]], batch_src[j,src_idx_s:src_idx_e_,stdim:], L2=False, GV=False)

                            tmp_batch_loss_lat_src = loss_vae(batch_lat_src[i][j,:flen_acc[j]], lat_dim=args.lat_dim)
                            tmp_batch_loss_lat_src_cv = loss_vae(batch_lat_src_trg[i][j,:flen_acc[j]], lat_dim=args.lat_dim)

                            if os.path.basename(os.path.dirname(featfile_src[j])) == args.spk_src:
                                loss_mcd_src_src[i].append(tmp_batch_loss_mcd_src_src.item())
                                loss_mcd_src_trg_src[i].append(tmp_batch_loss_mcd_src_trg_src.item())
                                loss_mcd_src_trg[i].append(tmp_batch_loss_mcd_src_trg.item())

                                loss_lat_src_cv[i].append(tmp_batch_loss_lat_src_cv.item())
                                loss_lat_src[i].append(tmp_batch_loss_lat_src.item())
                            else:
                                loss_mcd_trg_trg[i].append(tmp_batch_loss_mcd_src_src.item())
                                loss_mcd_trg_src_trg[i].append(tmp_batch_loss_mcd_src_trg_src.item())
                                loss_mcd_trg_src[i].append(tmp_batch_loss_mcd_src_trg.item())

                                loss_lat_trg_cv[i].append(tmp_batch_loss_lat_src_cv.item())
                                loss_lat_trg[i].append(tmp_batch_loss_lat_src.item())

                            if k > 0:
                                batch_loss_mcd_src_src[i] = torch.cat((batch_loss_mcd_src_src[i], tmp_batch_loss_mcd_src_src.unsqueeze(0)))
                                batch_loss_mcd_src_trg_src[i] = torch.cat((batch_loss_mcd_src_trg_src[i], tmp_batch_loss_mcd_src_trg_src.unsqueeze(0)))
                                batch_loss_mcd_src_trg[i] = torch.cat((batch_loss_mcd_src_trg[i], tmp_batch_loss_mcd_src_trg.unsqueeze(0)))
                                batch_loss_lat_src[i] = torch.cat((batch_loss_lat_src[i], tmp_batch_loss_lat_src.unsqueeze(0)))
                                batch_loss_lat_src_cv[i] = torch.cat((batch_loss_lat_src[i], tmp_batch_loss_lat_src_cv.unsqueeze(0)))
                            else:
                                batch_loss_mcd_src_src[i] = tmp_batch_loss_mcd_src_src.unsqueeze(0)
                                batch_loss_mcd_src_trg_src[i] = tmp_batch_loss_mcd_src_trg_src.unsqueeze(0)
                                batch_loss_mcd_src_trg[i] = tmp_batch_loss_mcd_src_trg.unsqueeze(0)
                                batch_loss_lat_src[i] = tmp_batch_loss_lat_src.unsqueeze(0)
                                batch_loss_lat_src_cv[i] = tmp_batch_loss_lat_src_cv.unsqueeze(0)

                        if i > 0: 
                            if not half_cyc:
                                batch_loss += batch_loss_mcd_src_src[i].sum() + batch_loss_mcd_src_trg_src[i].sum() + batch_loss_lat_src[i].sum() + batch_loss_lat_src_cv[i].sum()
                            else:
                                batch_loss += batch_loss_mcd_src_src[i].sum() + batch_loss_lat_src[i].sum()
                        else:
                            if not half_cyc:
                                batch_loss = batch_loss_mcd_src_src[0].sum() + batch_loss_mcd_src_trg_src[0].sum() + batch_loss_lat_src[0].sum() + batch_loss_lat_src_cv[0].sum()
                            else:
                                batch_loss = batch_loss_mcd_src_src[0].sum() + batch_loss_lat_src[0].sum()

                        batch_loss_mcd_src_src[i] = torch.mean(batch_loss_mcd_src_src[i])
                        batch_loss_mcd_src_trg_src[i] = torch.mean(batch_loss_mcd_src_trg_src[i])
                        batch_loss_mcd_src_trg[i] = torch.mean(batch_loss_mcd_src_trg[i])
                        batch_loss_lat_src[i] = torch.mean(batch_loss_lat_src[i])
                        batch_loss_lat_src_cv[i] = torch.mean(batch_loss_lat_src_cv[i])

                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                    loss.append(batch_loss.item())

                    print_mcd_flag = False
                    for i in range(args.n_cyc):
                        batch_mcdpow_src_src[i] = []
                        batch_mcd_src_src[i] = []
                        batch_mcdpow_src_trg_src[i] = []
                        batch_mcd_src_trg_src[i] = []

                    for j in select_utt_idx:
                        if spcidx_src_s_idx[j] >= 0:
                            print_mcd_flag = True
                            for i in range(args.n_cyc):
                                tmp_batch_mcdpow_src_src, _ = dtw.calc_mcd(np.array(torch.index_select(batch_src[j],0,spcidx_src[j,spcidx_src_s_idx[j]:spcidx_src_e_idx[j]+1])[:,stdim:].cpu().data.numpy(), dtype=np.float64), np.array(torch.index_select(batch_trj_src_src[i][j],0,spcidx_src[j,spcidx_src_s_idx[j]:spcidx_src_e_idx[j]+1]-src_idx_s).cpu().data.numpy(), dtype=np.float64))
                                tmp_batch_mcd_src_src, _ = dtw.calc_mcd(np.array(torch.index_select(batch_src[j],0,spcidx_src[j,spcidx_src_s_idx[j]:spcidx_src_e_idx[j]+1])[:,stdim_:].cpu().data.numpy(), dtype=np.float64), np.array(torch.index_select(batch_trj_src_src[i][j],0,spcidx_src[j,spcidx_src_s_idx[j]:spcidx_src_e_idx[j]+1]-src_idx_s)[:,1:].cpu().data.numpy(), dtype=np.float64))

                                tmp_batch_mcdpow_src_trg_src, _ = dtw.calc_mcd(np.array(torch.index_select(batch_src[j],0,spcidx_src[j,spcidx_src_s_idx[j]:spcidx_src_e_idx[j]+1])[:,stdim:].cpu().data.numpy(), dtype=np.float64), np.array(torch.index_select(batch_trj_src_trg_src[i][j],0,spcidx_src[j,spcidx_src_s_idx[j]:spcidx_src_e_idx[j]+1]-src_idx_s).cpu().data.numpy(), dtype=np.float64))
                                tmp_batch_mcd_src_trg_src, _ = dtw.calc_mcd(np.array(torch.index_select(batch_src[j],0,spcidx_src[j,spcidx_src_s_idx[j]:spcidx_src_e_idx[j]+1])[:,stdim_:].cpu().data.numpy(), dtype=np.float64), np.array(torch.index_select(batch_trj_src_trg_src[i][j],0,spcidx_src[j,spcidx_src_s_idx[j]:spcidx_src_e_idx[j]+1]-src_idx_s)[:,1:].cpu().data.numpy(), dtype=np.float64))
                              
                                batch_mcdpow_src_src[i].append(tmp_batch_mcdpow_src_src)
                                batch_mcd_src_src[i].append(tmp_batch_mcd_src_src)

                                batch_mcdpow_src_trg_src[i].append(tmp_batch_mcdpow_src_trg_src)
                                batch_mcd_src_trg_src[i].append(tmp_batch_mcd_src_trg_src)

                                if os.path.basename(os.path.dirname(featfile_src[j])) == args.spk_src:
                                    mcdpow_src_src[i].append(tmp_batch_mcdpow_src_src)
                                    mcd_src_src[i].append(tmp_batch_mcd_src_src)
    
                                    mcdpow_src_trg_src[i].append(tmp_batch_mcdpow_src_trg_src)
                                    mcd_src_trg_src[i].append(tmp_batch_mcd_src_trg_src)
                                else:
                                    mcdpow_trg_trg[i].append(tmp_batch_mcdpow_src_src)
                                    mcd_trg_trg[i].append(tmp_batch_mcd_src_src)
    
                                    mcdpow_trg_src_trg[i].append(tmp_batch_mcdpow_src_trg_src)
                                    mcd_trg_src_trg[i].append(tmp_batch_mcd_src_trg_src)
                           
                    text_log = "%.3f ;; " % batch_loss.item()
                    if print_mcd_flag:
                        for i in range(args.n_cyc):
                            text_log += "[%d] %.3f %.3f %.3f ; %.3f %.3f ; %.3f dB %.3f dB , %.3f dB %.3f dB ;; " % (
                                         i+1, batch_loss_mcd_src_src[i].item(), batch_loss_mcd_src_trg_src[i].item(), batch_loss_mcd_src_trg[i].item(),
                                             batch_loss_lat_src[i].item(), batch_loss_lat_src_cv[i].item(),
                                                 np.mean(batch_mcdpow_src_src[i]), np.mean(batch_mcd_src_src[i]), np.mean(batch_mcdpow_src_trg_src[i]), np.mean(batch_mcd_src_trg_src[i]))
                    else:
                        for i in range(args.n_cyc):
                            text_log += "[%d] %.3f %.3f %.3f ; %.3f %.3f ;; " % (
                                         i+1, batch_loss_mcd_src_src[i].item(), batch_loss_mcd_src_trg_src[i].item(), batch_loss_mcd_src_trg[i].item(),
                                             batch_loss_lat_src[i].item(), batch_loss_lat_src_cv[i].item())
                    logging.info("batch loss [%d] = %s  (%.3f sec)" % (c_idx_src+1, text_log, time.time() - start))
                    iter_idx += 1
                    iter_count += 1
                total.append(time.time() - start)
            else: # utterance-length mini-batch
                for i in range(n_batch_utt):
                    logging.info("%s %s %d %d %d %d" % (featfile_src[i], featfile_src_trg[i], flens_src[i], flens_src_trg[i], flens_spc_src[i], flens_spc_src_trg[i]))

                if n_batch_utt == args.batch_size_utt:
                    y_in_pp_ = y_in_pp
                    y_in_trg_ = y_in_trg
                    y_in_src_ = y_in_src
                else:
                    y_in_pp_ = y_in_pp_mod
                    y_in_trg_ = y_in_trg_mod
                    y_in_src_ = y_in_src_mod

                with torch.no_grad():
                    trj_lat_srctrg, _, _ = model_encoder(batch_src_trg, y_in_pp_, clamp_vae=True, lat_dim=args.lat_dim)
                for i in range(args.n_cyc):
                    batch_mcdpow_src_src[i] = []
                    batch_mcd_src_src[i] = []
                    batch_mcdpow_src_trg_src[i] = []
                    batch_mcd_src_trg_src[i] = []

                    if i > 0:
                        batch_lat_src[i], _, _ = model_encoder(torch.cat((batch_src[:,:,:stdim], batch_trj_src_trg_src[i-1]),2), y_in_pp_, clamp_vae=True, lat_dim=args.lat_dim, do=True)
                        batch_trj_src_src[i], _, _ = model_decoder(torch.cat((batch_src_src_code, sampling_vae_batch(batch_lat_src[i], lat_dim=args.lat_dim)),2), y_in_src_, do=True)
                        batch_trj_src_trg[i], _, _ = model_decoder(torch.cat((batch_src_trg_code, sampling_vae_batch(batch_lat_src[i], lat_dim=args.lat_dim)),2), y_in_trg_, do=True)

                        batch_lat_src_trg[i], _, _ = model_encoder(torch.cat((batch_cv_src, batch_trj_src_trg[i]),2), y_in_pp_, clamp_vae=True, lat_dim=args.lat_dim, do=True)
                        batch_trj_src_trg_src[i], _, _ = model_decoder(torch.cat((batch_src_src_code, sampling_vae_batch(batch_lat_src_trg[i], lat_dim=args.lat_dim)),2), y_in_src_, do=True)
                    else:
                        batch_lat_src[0], _, _ = model_encoder(batch_src, y_in_pp_, clamp_vae=True, lat_dim=args.lat_dim, do=True)
                        batch_trj_src_src[0], _, _ = model_decoder(torch.cat((batch_src_src_code, sampling_vae_batch(batch_lat_src[0], lat_dim=args.lat_dim)),2), y_in_src_, do=True)
                        batch_trj_src_trg[0], _, _ = model_decoder(torch.cat((batch_src_trg_code, sampling_vae_batch(batch_lat_src[0], lat_dim=args.lat_dim)),2), y_in_trg_, do=True)

                        batch_lat_src_trg[0], _, _ = model_encoder(torch.cat((batch_cv_src, batch_trj_src_trg[0]),2), y_in_pp_, clamp_vae=True, lat_dim=args.lat_dim, do=True)
                        batch_trj_src_trg_src[0], _, _ = model_decoder(torch.cat((batch_src_src_code, sampling_vae_batch(batch_lat_src_trg[0], lat_dim=args.lat_dim)),2), y_in_src_, do=True)

                        batch_mcdpow_src_trg[i] = []
                        batch_mcd_src_trg[i] = []
                        batch_lat_dist_srctrg1[i] = []
                        batch_lat_dist_srctrg2[i] = []
                        for j in range(n_batch_utt):
                            if os.path.basename(os.path.dirname(featfile_src[j])) == args.spk_src:
                                gv_src_src[i].append(np.var(batch_trj_src_src[i][j,:flens_src[j],1:].cpu().data.numpy(), axis=0))
                                gv_src_trg[i].append(np.var(batch_trj_src_trg[i][j,:flens_src[j],1:].cpu().data.numpy(), axis=0))
                                gv_src_trg_src[i].append(np.var(batch_trj_src_trg_src[i][j,:flens_src[j],1:].cpu().data.numpy(), axis=0))
                            else:
                                gv_trg_trg[i].append(np.var(batch_trj_src_src[i][j,:flens_src[j],1:].cpu().data.numpy(), axis=0))
                                gv_trg_src[i].append(np.var(batch_trj_src_trg[i][j,:flens_src[j],1:].cpu().data.numpy(), axis=0))
                                gv_trg_src_trg[i].append(np.var(batch_trj_src_trg_src[i][j,:flens_src[j],1:].cpu().data.numpy(), axis=0))

                            trj_lat_srctrg_ = np.array(torch.index_select(trj_lat_srctrg[j],0,spcidx_src_trg[j,:flens_spc_src_trg[j]]).cpu().data.numpy(), dtype=np.float64)
                            trj_lat_src_ = np.array(torch.index_select(batch_lat_src[0][j],0,spcidx_src[j,:flens_spc_src[j]]).cpu().data.numpy(), dtype=np.float64)
                            aligned_lat_srctrg1, _, _, _ = dtw.dtw_org_to_trg(trj_lat_src_, trj_lat_srctrg_)
                            tmp_batch_lat_dist_srctrg1 = np.mean(np.sqrt(np.mean((aligned_lat_srctrg1-trj_lat_srctrg_)**2, axis=0)))
                            _, _, tmp_batch_lat_cdist_srctrg1, _ = dtw.dtw_org_to_trg(trj_lat_srctrg_, trj_lat_src_, mcd=0)
                            aligned_lat_srctrg2, _, _, _ = dtw.dtw_org_to_trg(trj_lat_srctrg_, trj_lat_src_)
                            tmp_batch_lat_dist_srctrg2 = np.mean(np.sqrt(np.mean((aligned_lat_srctrg2-trj_lat_src_)**2, axis=0)))
                            _, _, tmp_batch_lat_cdist_srctrg2, _ = dtw.dtw_org_to_trg(trj_lat_src_, trj_lat_srctrg_, mcd=0)

                            tmp_batch_lat_dist_srctrg1 = (tmp_batch_lat_dist_srctrg1+tmp_batch_lat_dist_srctrg2)/2
                            tmp_batch_lat_dist_srctrg2 = (tmp_batch_lat_cdist_srctrg1+tmp_batch_lat_cdist_srctrg2)/2

                            batch_lat_dist_srctrg1[0].append(tmp_batch_lat_dist_srctrg1)
                            batch_lat_dist_srctrg2[0].append(tmp_batch_lat_dist_srctrg2)

                            _, _, tmp_batch_mcdpow_src_trg, _ = dtw.dtw_org_to_trg(np.array(torch.index_select(batch_trj_src_trg[i][j],0,spcidx_src[j,:flens_spc_src[j]]).cpu().data.numpy(), dtype=np.float64), np.array(torch.index_select(batch_src_trg[j,:,stdim:],0,spcidx_src_trg[j,:flens_spc_src_trg[j]]).cpu().data.numpy(), dtype=np.float64))
                            _, _, tmp_batch_mcd_src_trg, _ = dtw.dtw_org_to_trg(np.array(torch.index_select(batch_trj_src_trg[i][j,:,1:],0,spcidx_src[j,:flens_spc_src[j]]).cpu().data.numpy(), dtype=np.float64), np.array(torch.index_select(batch_src_trg[j,:,stdim_:],0,spcidx_src_trg[j,:flens_spc_src_trg[j]]).cpu().data.numpy(), dtype=np.float64))

                            batch_mcdpow_src_trg[0].append(tmp_batch_mcdpow_src_trg)
                            batch_mcd_src_trg[0].append(tmp_batch_mcd_src_trg)

                            text_log = "%s %s = %.3f dB %.3f dB , %.3f %.3f" % (
                                    featfile_src[j], featfile_src_trg[j], tmp_batch_mcdpow_src_trg, tmp_batch_mcd_src_trg, tmp_batch_lat_dist_srctrg1, tmp_batch_lat_dist_srctrg2)
                            if os.path.basename(os.path.dirname(featfile_src[j])) == args.spk_src:
                                mcdpow_src_trg[i].append(tmp_batch_mcdpow_src_trg)
                                mcd_src_trg[i].append(tmp_batch_mcd_src_trg)
                                lat_dist_srctrg1[0].append(tmp_batch_lat_dist_srctrg1)
                                lat_dist_srctrg2[0].append(tmp_batch_lat_dist_srctrg2)
                                logging.info("batch srctrg loss %s " % (text_log))
                            else:
                                mcdpow_trg_src[i].append(tmp_batch_mcdpow_src_trg)
                                mcd_trg_src[i].append(tmp_batch_mcd_src_trg)
                                lat_dist_trgsrc1[0].append(tmp_batch_lat_dist_srctrg1)
                                lat_dist_trgsrc2[0].append(tmp_batch_lat_dist_srctrg2)
                                logging.info("batch trgsrc loss %s " % (text_log))

                        batch_mcdpow_src_trg[i] = np.mean(batch_mcdpow_src_trg[i])
                        batch_mcd_src_trg[i] = np.mean(batch_mcd_src_trg[i])
                        batch_lat_dist_srctrg1[i] = np.mean(batch_lat_dist_srctrg1[i])
                        batch_lat_dist_srctrg2[i] = np.mean(batch_lat_dist_srctrg2[i])

                    for j in range(n_batch_utt):
                        _, tmp_batch_loss_mcd_src_src, _ = criterion_mcd(batch_trj_src_src[i][j,:flens_src[j]], batch_src[j,:flens_src[j],stdim:], L2=False, GV=False)
                        _, tmp_batch_loss_mcd_src_trg, _ = criterion_mcd(batch_trj_src_trg[i][j,:flens_src[j]], batch_src[j,:flens_src[j],stdim:], L2=False, GV=False)
                        _, tmp_batch_loss_mcd_src_trg_src, _ = criterion_mcd(batch_trj_src_trg_src[i][j,:flens_src[j]], batch_src[j,:flens_src[j],stdim:], L2=False, GV=False)

                        tmp_batch_loss_lat_src = loss_vae(batch_lat_src[i][j,:flens_src[j]], lat_dim=args.lat_dim)
                        tmp_batch_loss_lat_src_cv = loss_vae(batch_lat_src_trg[i][j,:flens_src[j]], lat_dim=args.lat_dim)

                        batch_src_spc_ = np.array(torch.index_select(batch_src[j,:,stdim:],0,spcidx_src[j,:flens_spc_src[j]]).cpu().data.numpy(), dtype=np.float64)
                        batch_src_spc__ = np.array(torch.index_select(batch_src[j,:,stdim_:],0,spcidx_src[j,:flens_spc_src[j]]).cpu().data.numpy(), dtype=np.float64)

                        tmp_batch_mcdpow_src_src, _ = dtw.calc_mcd(batch_src_spc_, np.array(torch.index_select(batch_trj_src_src[i][j],0,spcidx_src[j,:flens_spc_src[j]]).cpu().data.numpy(), dtype=np.float64))
                        tmp_batch_mcd_src_src, _ = dtw.calc_mcd(batch_src_spc__, np.array(torch.index_select(batch_trj_src_src[i][j,:,1:],0,spcidx_src[j,:flens_spc_src[j]]).cpu().data.numpy(), dtype=np.float64))

                        tmp_batch_mcdpow_src_trg_src, _ = dtw.calc_mcd(batch_src_spc_, np.array(torch.index_select(batch_trj_src_trg_src[i][j],0,spcidx_src[j,:flens_spc_src[j]]).cpu().data.numpy(), dtype=np.float64))
                        tmp_batch_mcd_src_trg_src, _ = dtw.calc_mcd(batch_src_spc__, np.array(torch.index_select(batch_trj_src_trg_src[i][j,:,1:],0,spcidx_src[j,:flens_spc_src[j]]).cpu().data.numpy(), dtype=np.float64))

                        if j > 0:
                            batch_loss_mcd_src_src[i] = torch.cat((batch_loss_mcd_src_src[i], tmp_batch_loss_mcd_src_src.unsqueeze(0)))
                            batch_loss_mcd_src_trg[i] = torch.cat((batch_loss_mcd_src_trg[i], tmp_batch_loss_mcd_src_trg.unsqueeze(0)))
                            batch_loss_mcd_src_trg_src[i] = torch.cat((batch_loss_mcd_src_trg_src[i], tmp_batch_loss_mcd_src_trg_src.unsqueeze(0)))

                            batch_loss_lat_src[i] = torch.cat((batch_loss_lat_src[i], tmp_batch_loss_lat_src.unsqueeze(0)))
                            batch_loss_lat_src_cv[i] = torch.cat((batch_loss_lat_src_cv[i], tmp_batch_loss_lat_src_cv.unsqueeze(0)))
                        else:
                            batch_loss_mcd_src_src[i] = tmp_batch_loss_mcd_src_src.unsqueeze(0)
                            batch_loss_mcd_src_trg[i] = tmp_batch_loss_mcd_src_trg.unsqueeze(0)
                            batch_loss_mcd_src_trg_src[i] = tmp_batch_loss_mcd_src_trg_src.unsqueeze(0)

                            batch_loss_lat_src[i] = tmp_batch_loss_lat_src.unsqueeze(0)
                            batch_loss_lat_src_cv[i] = tmp_batch_loss_lat_src_cv.unsqueeze(0)

                        if os.path.basename(os.path.dirname(featfile_src[j])) == args.spk_src:
                            mcdpow_src_src[i].append(tmp_batch_mcdpow_src_src)
                            mcd_src_src[i].append(tmp_batch_mcd_src_src)
                            mcdpow_src_trg_src[i].append(tmp_batch_mcdpow_src_trg_src)
                            mcd_src_trg_src[i].append(tmp_batch_mcd_src_trg_src)

                            loss_mcd_src_src[i].append(tmp_batch_loss_mcd_src_src.item())
                            loss_mcd_src_trg[i].append(tmp_batch_loss_mcd_src_trg.item())
                            loss_mcd_src_trg_src[i].append(tmp_batch_loss_mcd_src_trg_src.item())

                            loss_lat_src[i].append(tmp_batch_loss_lat_src.item())
                            loss_lat_src_cv[i].append(tmp_batch_loss_lat_src_cv.item())
                        else:
                            mcdpow_trg_trg[i].append(tmp_batch_mcdpow_src_src)
                            mcd_trg_trg[i].append(tmp_batch_mcd_src_src)
                            mcdpow_trg_src_trg[i].append(tmp_batch_mcdpow_src_trg_src)
                            mcd_trg_src_trg[i].append(tmp_batch_mcd_src_trg_src)

                            loss_mcd_trg_trg[i].append(tmp_batch_loss_mcd_src_src.item())
                            loss_mcd_trg_src[i].append(tmp_batch_loss_mcd_src_trg.item())
                            loss_mcd_trg_src_trg[i].append(tmp_batch_loss_mcd_src_trg_src.item())

                            loss_lat_trg[i].append(tmp_batch_loss_lat_src.item())
                            loss_lat_trg_cv[i].append(tmp_batch_loss_lat_src_cv.item())

                        batch_mcdpow_src_src[i].append(tmp_batch_mcdpow_src_src)
                        batch_mcd_src_src[i].append(tmp_batch_mcd_src_src)
                        batch_mcdpow_src_trg_src[i].append(tmp_batch_mcdpow_src_trg_src)
                        batch_mcd_src_trg_src[i].append(tmp_batch_mcd_src_trg_src)

                    batch_mcdpow_src_src[i] = np.mean(batch_mcdpow_src_src[i])
                    batch_mcd_src_src[i] = np.mean(batch_mcd_src_src[i])
                    batch_mcdpow_src_trg_src[i] = np.mean(batch_mcdpow_src_trg_src[i])
                    batch_mcd_src_trg_src[i] = np.mean(batch_mcd_src_trg_src[i])

                    if i > 0: 
                        if not half_cyc:
                            batch_loss += batch_loss_mcd_src_src[i].sum() + batch_loss_mcd_src_trg_src[i].sum() + batch_loss_lat_src[i].sum() + batch_loss_lat_src_cv[i].sum()
                        else:
                            batch_loss += batch_loss_mcd_src_src[i].sum() + batch_loss_lat_src[i].sum()
                    else:
                        if not half_cyc:
                            batch_loss = batch_loss_mcd_src_src[0].sum() + batch_loss_mcd_src_trg_src[0].sum() + batch_loss_lat_src[0].sum() + batch_loss_lat_src_cv[0].sum()
                        else:
                            batch_loss = batch_loss_mcd_src_src[0].sum() + batch_loss_lat_src[0].sum()

                    batch_loss_mcd_src_src[i] = torch.mean(batch_loss_mcd_src_src[i])
                    batch_loss_mcd_src_trg_src[i] = torch.mean(batch_loss_mcd_src_trg_src[i])
                    batch_loss_mcd_src_trg[i] = torch.mean(batch_loss_mcd_src_trg[i])
                    batch_loss_lat_src[i] = torch.mean(batch_loss_lat_src[i])
                    batch_loss_lat_src_cv[i] = torch.mean(batch_loss_lat_src_cv[i])

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss.append(batch_loss.item())

                text_log = "%.3f ;; " % batch_loss.item()
                for i in range(args.n_cyc):
                    text_log += "[%d] %.3f %.3f %.3f ; %.3f %.3f ; %.3f dB %.3f dB , %.3f dB %.3f dB;; " % (
                                 i+1, batch_loss_mcd_src_src[i].item(), batch_loss_mcd_src_trg_src[i].item(), batch_loss_mcd_src_trg[i].item(),
                                     batch_loss_lat_src[i].item(), batch_loss_lat_src_cv[i].item(), batch_mcdpow_src_src[i], batch_mcd_src_src[i],
                                         batch_mcdpow_src_trg_src[i], batch_mcd_src_trg_src[i])
                logging.info("batch loss [%d] = %s  (%.3f sec)" % (c_idx_src+1, text_log, time.time() - start))
                iter_idx += 1
                iter_count += 1
                total.append(time.time() - start)

    # save final model
    model_encoder.cpu()
    model_decoder.cpu()
    torch.save({"model_encoder": model_encoder.state_dict(), "model_decoder": model_decoder.state_dict()}, args.expdir + "/checkpoint-final.pkl")
    logging.info("final checkpoint created.")


if __name__ == "__main__":
    main()
