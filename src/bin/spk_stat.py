#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate histograms to decide speaker-dependent parameters

"""

import argparse
import os
from pathlib import Path
import logging

import matplotlib
import numpy as np
from utils import check_hdf5
from utils import read_hdf5
from utils import read_txt
from utils import write_hdf5

matplotlib.use('Agg')  # noqa #isort:skip
import matplotlib.pyplot as plt  # isort:skip


def create_histogram(data, figure_path, range_min=-70, range_max=20,
                     step=10, n_bins=200, xlabel='Power [dB]'):
    """Create histogram

    Parameters
    ----------
    data : list,
        List of several data sequences
    figure_path : str,
        Filepath to be output figure
    range_min : int, optional,
        Minimum range for histogram
        Default set to -70
    range_max : int, optional,
        Maximum range for histogram
        Default set to -20
    step : int, optional
        Stap size of label in horizontal axis
        Default set to 10
    xlabel : str, optional
        Label of the horizontal axis
        Default set to 'Power [dB]'

    """

    # plot histgram
    #plt.hist(data, bins=200, range=(range_min, range_max),
    #         density=True, histtype="stepfilled")
    hist, bins, _ = plt.hist(data, bins=n_bins, range=(range_min, range_max),
             density=True, histtype="stepfilled")
    plt.xlabel(xlabel)
    plt.ylabel("Probability")
    plt.xticks(np.arange(range_min, range_max, step))

    figure_dir = os.path.dirname(figure_path)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    plt.savefig(figure_path)
    plt.close()

    return hist, bins


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feats", default=None, required=True,
        help="name of the list of hdf5 files")
    parser.add_argument("--expdir", required=True,
        type=str, help="directory to save the log")
    parser.add_argument("--spkr", default=None,
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
                            filename=args.expdir + "/spk_stat.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/spk_stat.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/spk_stat.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    filenames = read_txt(args.feats)
    logging.info("number of training utterances = %d" % len(filenames))

    npows = np.empty((0))
    f0s = np.empty((0))
    # process over all of data
    for filename in filenames:
        logging.info(filename)
        f0 = read_hdf5(filename, "/f0")
        npow = read_hdf5(filename, "/npow")
        nonzero_indices = np.nonzero(f0)
        logging.info(f0[nonzero_indices].shape)
        logging.info(f0s.shape)
        f0s = np.concatenate([f0s,f0[nonzero_indices]])
        logging.info(f0s.shape)
        logging.info(npows.shape)
        npows = np.concatenate([npows,npow])
        logging.info(npows.shape)

    # create a histogram to visualize F0 range of the speaker
    f0histogrampath = os.path.join(
        args.expdir, args.spkr + '_f0histogram.png')
    f0hist, f0bins = create_histogram(f0s, f0histogrampath, range_min=40, range_max=700,
                     step=50, n_bins=660, xlabel='Fundamental frequency [Hz]')
    f0histogrampath = os.path.join(
        args.expdir, args.spkr + '_f0histogram.txt')
    f = open(f0histogrampath, 'w')
    for i in range(f0hist.shape[0]):
        f.write('%d %.9f\n' % (f0bins[i], f0hist[i]))
    f.close()

    # create a histogram to visualize npow range of the speaker
    npowhistogrampath = os.path.join(
        args.expdir, args.spkr + '_npowhistogram.png')
    npowhist, npowbins = create_histogram(npows, npowhistogrampath, range_min=-70, range_max=20,
                     step=10, n_bins=180, xlabel="Frame power [dB]")
    npowhistogrampath = os.path.join(
        args.expdir, args.spkr + '_npowhistogram.txt')
    f = open(npowhistogrampath, 'w')
    for i in range(npowhist.shape[0]):
        f.write('%.1f %.9f\n' % (npowbins[i], npowhist[i]))
    f.close()


if __name__ == '__main__':
    main()
