#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
# based on PyTorch implementation for WaveNet vocoder by Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function

import argparse
import logging

import numpy as np
from sklearn.preprocessing import StandardScaler

from utils import check_hdf5
from utils import read_hdf5
from utils import read_txt
from utils import write_hdf5
from distutils.util import strtobool


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feats_src", default=None, required=True,
        help="name of the list of hdf5 files")
    parser.add_argument(
        "--feats_trg", default=None, required=True,
        help="name of the list of hdf5 files")
    parser.add_argument(
        "--feats_trg_all", default=None,
        help="name of the list of hdf5 files")
    parser.add_argument(
        "--stats", default=None, required=True,
        help="filename of hdf5 format")
    parser.add_argument(
        "--stats_trg", default=None,
        help="filename of hdf5 format")
    parser.add_argument("--expdir", required=True,
        type=str, help="directory to save the log")
    parser.add_argument(
        "--verbose", default=1,
        type=int, help="log message level")

    args = parser.parse_args()

    # set log level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/calc_stats.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/calc_stats.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/calc_stats.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    # define scaler
    #scaler_sdfeatuvcat_range = StandardScaler()
    scaler_feat_org_lf0_jnt = StandardScaler()
    if args.feats_trg_all is not None:
        scaler_feat_org_lf0_trg_jnt = StandardScaler()

    # read source list
    filenames = read_txt(args.feats_src)
    print("number of source training utterances =", len(filenames))

    for filename in filenames:
        #sdfeatuv_cat_range = read_hdf5(filename, "/sdfeat_uv_cat_range")
        #scaler_sdfeatuvcat_range.partial_fit(sdfeatuv_cat_range[:, :])
        feat_org_lf0 = read_hdf5(filename, "/feat_org_lf0")
        scaler_feat_org_lf0_jnt.partial_fit(feat_org_lf0[:, :])

    # read target list
    filenames = read_txt(args.feats_trg)
    print("number of target training utterances =", len(filenames))

    for filename in filenames:
        #sdfeatuv_cat_range = read_hdf5(filename, "/sdfeat_uv_cat_range")
        #scaler_sdfeatuvcat_range.partial_fit(sdfeatuv_cat_range[:, :])
        feat_org_lf0 = read_hdf5(filename, "/feat_org_lf0")
        scaler_feat_org_lf0_jnt.partial_fit(feat_org_lf0[:, :])
        if args.feats_trg_all is not None:
            scaler_feat_org_lf0_trg_jnt.partial_fit(feat_org_lf0[:, :])

    if args.feats_trg_all is not None:
        # read target all list
        filenames = read_txt(args.feats_trg_all)
        print("number of target all training utterances =", len(filenames))

        for filename in filenames:
            #sdfeatuv_cat_range = read_hdf5(filename, "/sdfeat_uv_cat_range")
            #scaler_sdfeatuvcat_range.partial_fit(sdfeatuv_cat_range[:, :])
            feat_org_lf0 = read_hdf5(filename, "/feat_org_lf0")
            scaler_feat_org_lf0_jnt.partial_fit(feat_org_lf0[:, :])
            scaler_feat_org_lf0_trg_jnt.partial_fit(feat_org_lf0[:, :])

    #mean_sdfeatuvcat_range = scaler_sdfeatuvcat_range.mean_
    #scale_sdfeatuvcat_range = scaler_sdfeatuvcat_range.scale_
    mean_feat_org_lf0_jnt = scaler_feat_org_lf0_jnt.mean_
    scale_feat_org_lf0_jnt = scaler_feat_org_lf0_jnt.scale_
    if args.feats_trg_all is not None:
        mean_feat_org_lf0_trg_jnt = scaler_feat_org_lf0_trg_jnt.mean_
        scale_feat_org_lf0_trg_jnt = scaler_feat_org_lf0_trg_jnt.scale_

    # write to hdf5
    #write_hdf5(args.stats, "/mean_sdfeat_uv_cat_range", mean_sdfeatuvcat_range)
    #write_hdf5(args.stats, "/scale_sdfeat_uv_cat_range", scale_sdfeatuvcat_range)
    print(mean_feat_org_lf0_jnt)
    print(scale_feat_org_lf0_jnt)
    if args.feats_trg_all is not None:
        print(mean_feat_org_lf0_trg_jnt)
        print(scale_feat_org_lf0_trg_jnt)
    write_hdf5(args.stats, "/mean_feat_org_lf0_jnt", mean_feat_org_lf0_jnt)
    write_hdf5(args.stats, "/scale_feat_org_lf0_jnt", scale_feat_org_lf0_jnt)
    if args.feats_trg_all is not None:
        write_hdf5(args.stats_trg, "/mean_feat_org_lf0_trg_jnt", mean_feat_org_lf0_trg_jnt)
        write_hdf5(args.stats_trg, "/scale_feat_org_lf0_trg_jnt", scale_feat_org_lf0_trg_jnt)


if __name__ == "__main__":
    main()
