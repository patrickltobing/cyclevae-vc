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

from feature_extract_vc import convert_f0, convert_continuos_f0, low_pass_filter

#FS = 16000
FS = 22050
#FS = 24000
#FS = 44100
#FS = 48000
SHIFTMS = 5.0

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feats_src", default=None, required=True,
        help="name of the list of hdf5 files")
    parser.add_argument(
        "--feats_trg", default=None, required=True,
        help="name of the list of hdf5 files")
    parser.add_argument(
        "--stats_src", default=None, required=True,
        help="filename of hdf5 format")
    parser.add_argument(
        "--stats_trg", default=None, required=True,
        help="filename of hdf5 format")
    parser.add_argument(
        "--fs", default=FS, type=int,
        help="mcep dimension")
    parser.add_argument("--expdir", required=True,
        type=str, help="directory to save the log")
    parser.add_argument("--spkr_src", default=None,
        type=str, help="directory to save the log")
    parser.add_argument("--spkr_trg", default=None,
        type=str, help="directory to save the log")
    parser.add_argument("--cv_src", default=True,
                        type=strtobool, help="flag to write gv stats")
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

    # read list and define scaler
    if args.cv_src:
        filenames_src = read_txt(args.feats_src)
    filenames_trg = read_txt(args.feats_trg)
    #assert(len(filenames_src) == len(filenames_trg))
    if args.cv_src:
        print("number of src utterances =", len(filenames_src))
    print("number of trg utterances =", len(filenames_trg))

    f0_range_mean_src = read_hdf5(args.stats_src, "/lf0_range_mean")
    f0_range_std_src = read_hdf5(args.stats_src, "/lf0_range_std")
    logging.info(f0_range_mean_src)
    logging.info(f0_range_std_src)
    f0_range_mean_trg = read_hdf5(args.stats_trg, "/lf0_range_mean")
    f0_range_std_trg = read_hdf5(args.stats_trg, "/lf0_range_std")
    logging.info(f0_range_mean_trg)
    logging.info(f0_range_std_trg)

    logging.info(args.fs)
    if args.fs == 44100:
        stdim = 2
        endim = 7
    elif args.fs == 22050:
        stdim = 2
        endim = 4
    elif args.fs == 24000:
        stdim = 2
        endim = 5
    elif args.fs == 48000:
        stdim = 2
        endim = 8
    else:
        stdim = 2
        endim = 4

    # process over all of source data
    if args.cv_src:
        for filename in filenames_src:
            logging.info(filename)
            ap = read_hdf5(filename, "/feat_org_lf0")[:,stdim:endim]
            f0 = read_hdf5(filename, "/f0_range")
            if (args.spkr_src is None and args.spkr_trg is None) or (args.spkr_src is not None and args.spkr_src in filename):
                cvf0 = convert_f0(f0, f0_range_mean_src, f0_range_std_src, f0_range_mean_trg, f0_range_std_trg)
            elif args.spkr_trg is not None and args.spkr_trg in filename:
                cvf0 = convert_f0(f0, f0_range_mean_trg, f0_range_std_trg, f0_range_mean_src, f0_range_std_src)
            cvuv, cont_f0 = convert_continuos_f0(cvf0)
            cvuv = np.expand_dims(cvuv, axis=-1)
            cont_f0_lpf = low_pass_filter(cont_f0, int(1.0 / (SHIFTMS * 0.001)), cutoff=20)
            cvlogf0fil = np.expand_dims(np.log(cont_f0_lpf), axis=-1)
            write_hdf5(filename, "/cvuvlogf0fil_ap", np.c_[cvuv,cvlogf0fil,ap])

    # process over all of target data
    for filename in filenames_trg:
        logging.info(filename)
        ap = read_hdf5(filename, "/feat_org_lf0")[:,stdim:endim]
        f0 = read_hdf5(filename, "/f0_range")
        if (args.spkr_trg is None and args.spkr_src is None) or (args.spkr_trg is not None and args.spkr_trg in filename):
            cvf0 = convert_f0(f0, f0_range_mean_trg, f0_range_std_trg, f0_range_mean_src, f0_range_std_src)
        elif args.spkr_src is not None and args.spkr_src in filename:
            cvf0 = convert_f0(f0, f0_range_mean_src, f0_range_std_src, f0_range_mean_trg, f0_range_std_trg)
        cvuv, cont_f0 = convert_continuos_f0(cvf0)
        cvuv = np.expand_dims(cvuv, axis=-1)
        cont_f0_lpf = low_pass_filter(cont_f0, int(1.0 / (SHIFTMS * 0.001)), cutoff=20)
        cvlogf0fil = np.expand_dims(np.log(cont_f0_lpf), axis=-1)
        write_hdf5(filename, "/cvuvlogf0fil_ap", np.c_[cvuv,cvlogf0fil,ap])


if __name__ == "__main__":
    main()
