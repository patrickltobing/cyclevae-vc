#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
# based on PyTorch implementation for VC by Kazuhiro Kobayashi (Nagoya University)
#
# Distributed under terms of the MIT license.

"""

"""

import numpy as np
import torch
import os
import logging
from utils import read_hdf5
from torch.utils.data import Dataset
import soundfile as sf



def padding(x, flen, value=0):
    """Pad values to end by flen"""
    diff = flen - x.shape[0]
    if diff > 0:
        if len(x.shape) > 1:
            x = np.concatenate([x, np.ones((diff, x.shape[1])) * value])
        else:
            x = np.concatenate([x, np.ones(diff) * value])
    return x


class FeatureDatasetInit(Dataset):
    """Dataset for init
    """

    def __init__(self, file_list, pad_transform):
        self.file_list = file_list
        self.pad_transform = pad_transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        featfile = self.file_list[idx]

        h = torch.FloatTensor(self.pad_transform(read_hdf5(featfile, "/feat_org_lf0")))
        flen = h.shape[0]

        return {'h': h, 'flen': flen, 'featfile': featfile}


class FeatureDatasetSingleVAE(Dataset):
    """Dataset for one-to-one
    """

    def __init__(self, file_list_src, file_list_src_trg, pad_transform, spk_src):
        self.file_list_src = file_list_src
        self.file_list_src_trg = file_list_src_trg
        self.pad_transform = pad_transform
        self.spk_src = spk_src

    def __len__(self):
        return len(self.file_list_src)

    def __getitem__(self, idx):
        featfile_src = self.file_list_src[idx]
        featfile_src_trg = self.file_list_src_trg[idx]

        h_src = read_hdf5(featfile_src, "/feat_org_lf0")
        flen_src = h_src.shape[0]
        src_code = np.zeros((flen_src,2))
        trg_code = np.zeros((flen_src,2))
        if os.path.basename(os.path.dirname(featfile_src)) == self.spk_src:
            src_code[:,0] = 1
            trg_code[:,1] = 1
        else:
            src_code[:,1] = 1
            trg_code[:,0] = 1
        cv_src = read_hdf5(featfile_src, "/cvuvlogf0fil_ap")
        spcidx_src = read_hdf5(featfile_src, "/spcidx_range")[0]
        flen_spc_src = spcidx_src.shape[0]
        h_src_trg = read_hdf5(featfile_src_trg, "/feat_org_lf0")
        flen_src_trg = h_src_trg.shape[0]
        spcidx_src_trg = read_hdf5(featfile_src_trg, "/spcidx_range")[0]
        flen_spc_src_trg = spcidx_src_trg.shape[0]

        h_src = torch.FloatTensor(self.pad_transform(h_src))
        src_code = torch.FloatTensor(self.pad_transform(src_code))
        trg_code = torch.FloatTensor(self.pad_transform(trg_code))
        cv_src = torch.FloatTensor(self.pad_transform(cv_src))
        spcidx_src = torch.LongTensor(self.pad_transform(spcidx_src))
        h_src_trg = torch.FloatTensor(self.pad_transform(h_src_trg))
        spcidx_src_trg = torch.LongTensor(self.pad_transform(spcidx_src_trg))

        return {'h_src': h_src, 'flen_src': flen_src, 'src_code': src_code, 'trg_code': trg_code, 'cv_src': cv_src, 'spcidx_src': spcidx_src, 'flen_spc_src': flen_spc_src, \
                'h_src_trg': h_src_trg, 'flen_src_trg': flen_src_trg, 'spcidx_src_trg': spcidx_src_trg, 'flen_spc_src_trg': flen_spc_src_trg, 'featfile_src': featfile_src, 'featfile_src_trg': featfile_src_trg}


def proc_multspk_data_random(featfile, spk_src_list, spk_trg_list, n_cyc, src_code):
    featfile_spk = os.path.basename(os.path.dirname(featfile))
    flag_src = False
    for i in range(len(spk_src_list)):
        if featfile_spk == spk_src_list[i]:
            src_code[:,i] = 1
            flag_src = True
            break
    if not flag_src:
        for i in range(len(spk_trg_list)):
            if featfile_spk == spk_trg_list[i]:
                src_code[:,i+len(spk_src_list)] = 1
                break
    cv_src_list = [None]*n_cyc
    trg_code_list = [None]*n_cyc
    pair_spk_list = [None]*n_cyc
    if flag_src:
        for i in range(n_cyc):
            trg_code_list[i] = np.zeros((src_code.shape[0],src_code.shape[1]))
            pair_idx = np.random.randint(0,len(spk_trg_list))
            trg_code_list[i][:,pair_idx+len(spk_src_list)] = 1
            pair_spk = spk_trg_list[pair_idx]
            cv_src_list[i] = read_hdf5(featfile, "/cvuvlogf0fil_ap_"+pair_spk)
            pair_spk_list[i] = pair_spk
    else:
        for i in range(n_cyc):
            trg_code_list[i] = np.zeros((src_code.shape[0],src_code.shape[1]))
            pair_idx = np.random.randint(0,len(spk_src_list))
            trg_code_list[i][:,pair_idx] = 1
            pair_spk = spk_src_list[pair_idx]
            cv_src_list[i] = read_hdf5(featfile, "/cvuvlogf0fil_ap_"+pair_spk)
            pair_spk_list[i] = pair_spk
    featfile_src_trg = os.path.dirname(os.path.dirname(featfile))+"/"+pair_spk_list[0]+"/"+os.path.basename(featfile)

    return cv_src_list, trg_code_list, featfile_spk, featfile_src_trg, pair_spk_list


class FeatureDatasetMultTrainVAE(Dataset):
    """Dataset for training many-to-many
    """

    def __init__(self, file_list, pad_transform, spk_src_list, spk_trg_list, n_cyc):
        self.file_list = file_list
        self.pad_transform = pad_transform
        self.spk_src_list = spk_src_list
        self.spk_trg_list = spk_trg_list
        self.n_spk_src = len(self.spk_src_list)
        self.n_spk_trg = len(self.spk_trg_list)
        self.n_spk = self.n_spk_src + self.n_spk_trg
        self.n_cyc = n_cyc

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        featfile_src = self.file_list[idx]

        h_src = read_hdf5(featfile_src, "/feat_org_lf0")
        flen_src = h_src.shape[0]
        src_code = np.zeros((flen_src,self.n_spk))

        cv_src_list, src_trg_code_list, featfile_spk, featfile_src_trg, pair_spk_list = proc_multspk_data_random(featfile_src, self.spk_src_list, self.spk_trg_list, self.n_cyc, src_code)

        spcidx_src = read_hdf5(featfile_src, "/spcidx_range")[0]
        flen_spc_src = spcidx_src.shape[0]

        h_src_trg = read_hdf5(featfile_src_trg, "/feat_org_lf0")
        flen_src_trg = h_src_trg.shape[0]
        spcidx_src_trg = read_hdf5(featfile_src_trg, "/spcidx_range")[0]
        flen_spc_src_trg = spcidx_src_trg.shape[0]

        h_src = torch.FloatTensor(self.pad_transform(h_src))
        src_code = torch.FloatTensor(self.pad_transform(src_code))

        for i in range(self.n_cyc):
            cv_src_list[i] = torch.FloatTensor(self.pad_transform(cv_src_list[i]))
            src_trg_code_list[i] = torch.FloatTensor(self.pad_transform(src_trg_code_list[i]))

        spcidx_src = torch.LongTensor(self.pad_transform(spcidx_src))

        h_src_trg = torch.FloatTensor(self.pad_transform(h_src_trg))
        spcidx_src_trg = torch.LongTensor(self.pad_transform(spcidx_src_trg))

        return {'h_src': h_src, 'flen_src': flen_src, 'src_code': src_code, 'src_trg_code_list': src_trg_code_list, 'cv_src_list': cv_src_list, 'spcidx_src': spcidx_src, 'flen_spc_src': flen_spc_src, \
                'h_src_trg': h_src_trg, 'flen_src_trg': flen_src_trg, 'spcidx_src_trg': spcidx_src_trg, 'flen_spc_src_trg': flen_spc_src_trg, 'featfile_src': featfile_src, 'featfile_src_trg': featfile_src_trg, \
                'featfile_spk': featfile_spk, 'pair_spk_list': pair_spk_list}


class FeatureDatasetMultEvalVAE(Dataset):
    """Dataset for evaluation many-to-many
    """

    def __init__(self, file_list_src_list, file_list_trg_list, pad_transform, spk_src_list, spk_trg_list):
        self.file_list_src_list = file_list_src_list
        self.file_list_trg_list = file_list_trg_list
        self.pad_transform = pad_transform
        self.spk_src_list = spk_src_list
        self.spk_trg_list = spk_trg_list
        self.n_spk_src = len(self.spk_src_list)
        self.n_spk_trg = len(self.spk_trg_list)
        self.n_spk = self.n_spk_src + self.n_spk_trg
        self.n_eval_utt = len(self.file_list_src_list[0])
        self.file_list_src = []
        self.file_list_src_trg = []
        self.count_spk_pair_cv = {}
        for i in range(self.n_spk_src):
            self.count_spk_pair_cv[self.spk_src_list[i]] = {}
            for j in range(self.n_spk_trg):
                self.count_spk_pair_cv[self.spk_src_list[i]][self.spk_trg_list[j]] = 0
        # deterministically select a conv. pair for each validation utterance
        if self.n_spk_trg > 1:
            idx_even_trg = 1
        else:
            idx_even_trg = 0
        idx_odd_trg = 0
        for spk_src_idx in range(self.n_spk_src):
            if spk_src_idx%2 == 0:
                if idx_even_trg >= self.n_spk_trg:
                    if self.n_spk_trg > 1:
                        idx_even_trg = 1
                    else:
                        idx_even_trg = 0
                spk_trg_idx = idx_even_trg
                idx_even_trg += 2
            else:
                if idx_odd_trg >= self.n_spk_trg:
                    idx_odd_trg = 0
                spk_trg_idx = idx_odd_trg
                idx_odd_trg += 2
            for i in range(self.n_eval_utt):
                self.count_spk_pair_cv[self.spk_src_list[spk_src_idx]][self.spk_trg_list[spk_trg_idx]] += 1
                self.file_list_src.append(self.file_list_src_list[spk_src_idx][i])
                self.file_list_src_trg.append(self.file_list_trg_list[spk_trg_idx][i])

    def __len__(self):
        return len(self.file_list_src)

    def __getitem__(self, idx):
        featfile_src = self.file_list_src[idx]
        featfile_src_trg = self.file_list_src_trg[idx]

        spk_src = os.path.basename(os.path.dirname(featfile_src))
        spk_trg = os.path.basename(os.path.dirname(featfile_src_trg))
        for i in range(self.n_spk_src):
            if spk_src == self.spk_src_list[i]:
                idx_src = i
                break
        for i in range(self.n_spk_trg):
            if spk_trg == self.spk_trg_list[i]:
                idx_trg = self.n_spk_src+i
                break

        h_src = read_hdf5(featfile_src, "/feat_org_lf0")
        flen_src = h_src.shape[0]
        src_code = np.zeros((flen_src,self.n_spk))
        src_trg_code = np.zeros((flen_src,self.n_spk))
        src_code[:,idx_src] = 1
        src_trg_code[:,idx_trg] = 1
        cv_src = read_hdf5(featfile_src, "/cvuvlogf0fil_ap_"+spk_trg)
        spcidx_src = read_hdf5(featfile_src, "/spcidx_range")[0]
        flen_spc_src = spcidx_src.shape[0]

        h_src_trg = read_hdf5(featfile_src_trg, "/feat_org_lf0")
        flen_src_trg = h_src_trg.shape[0]
        trg_code = np.zeros((flen_src_trg,self.n_spk))
        trg_src_code = np.zeros((flen_src_trg,self.n_spk))
        trg_code[:,idx_trg] = 1
        trg_src_code[:,idx_src] = 1
        cv_trg = read_hdf5(featfile_src_trg, "/cvuvlogf0fil_ap_"+spk_src)
        spcidx_src_trg = read_hdf5(featfile_src_trg, "/spcidx_range")[0]
        flen_spc_src_trg = spcidx_src_trg.shape[0]

        h_src = torch.FloatTensor(self.pad_transform(h_src))
        src_code = torch.FloatTensor(self.pad_transform(src_code))
        src_trg_code = torch.FloatTensor(self.pad_transform(src_trg_code))
        cv_src = torch.FloatTensor(self.pad_transform(cv_src))
        spcidx_src = torch.LongTensor(self.pad_transform(spcidx_src))

        h_src_trg = torch.FloatTensor(self.pad_transform(h_src_trg))
        trg_code = torch.FloatTensor(self.pad_transform(trg_code))
        trg_src_code = torch.FloatTensor(self.pad_transform(trg_src_code))
        cv_trg = torch.FloatTensor(self.pad_transform(cv_trg))
        spcidx_src_trg = torch.LongTensor(self.pad_transform(spcidx_src_trg))

        return {'h_src': h_src, 'flen_src': flen_src, 'src_code': src_code, 'src_trg_code': trg_code, 'cv_src': cv_src, 'spcidx_src': spcidx_src, 'flen_spc_src': flen_spc_src, \
                'h_src_trg': h_src_trg, 'flen_src_trg': flen_src_trg, 'trg_code': trg_code, 'trg_src_code': trg_src_code, 'cv_trg': cv_trg, 'spcidx_src_trg': spcidx_src_trg, \
                'flen_spc_src_trg': flen_spc_src_trg, 'featfile_src': featfile_src, 'featfile_src_trg': featfile_src_trg}


def proc_multspk_data_random_cls(featfile, spk_src_list, spk_trg_list, n_cyc, src_code):
    featfile_spk = os.path.basename(os.path.dirname(featfile))
    flag_src = False
    for i in range(len(spk_src_list)):
        if featfile_spk == spk_src_list[i]:
            src_code[:,i] = 1
            src_class_code = np.ones(src_code.shape[0],dtype=np.int64)*i
            flag_src = True
            break
    if not flag_src:
        for i in range(len(spk_trg_list)):
            if featfile_spk == spk_trg_list[i]:
                src_code[:,i+len(spk_src_list)] = 1
                src_class_code = np.ones(src_code.shape[0],dtype=np.int64)*(i+len(spk_src_list))
                break
    cv_src_list = [None]*n_cyc
    trg_code_list = [None]*n_cyc
    pair_spk_list = [None]*n_cyc
    trg_class_code_list = [None]*n_cyc
    if flag_src:
        for i in range(n_cyc):
            trg_code_list[i] = np.zeros((src_code.shape[0],src_code.shape[1]))
            pair_idx = np.random.randint(0,len(spk_trg_list))
            trg_code_list[i][:,pair_idx+len(spk_src_list)] = 1
            trg_class_code_list[i] = np.ones(src_code.shape[0],dtype=np.int64)*(pair_idx+len(spk_src_list))
            pair_spk = spk_trg_list[pair_idx]
            cv_src_list[i] = read_hdf5(featfile, "/cvuvlogf0fil_ap_"+pair_spk)
            pair_spk_list[i] = pair_spk
    else:
        for i in range(n_cyc):
            trg_code_list[i] = np.zeros((src_code.shape[0],src_code.shape[1]))
            pair_idx = np.random.randint(0,len(spk_src_list))
            trg_code_list[i][:,pair_idx] = 1
            trg_class_code_list[i] = np.ones(src_code.shape[0],dtype=np.int64)*pair_idx
            pair_spk = spk_src_list[pair_idx]
            cv_src_list[i] = read_hdf5(featfile, "/cvuvlogf0fil_ap_"+pair_spk)
            pair_spk_list[i] = pair_spk
    featfile_src_trg = os.path.dirname(os.path.dirname(featfile))+"/"+pair_spk_list[0]+"/"+os.path.basename(featfile)

    return cv_src_list, trg_code_list, featfile_spk, featfile_src_trg, pair_spk_list, src_class_code, trg_class_code_list


class FeatureDatasetMultTrainVAECls(Dataset):
    """Dataset for training many-to-many with classifier
    """

    def __init__(self, file_list, pad_transform, spk_src_list, spk_trg_list, n_cyc):
        self.file_list = file_list
        self.pad_transform = pad_transform
        self.spk_src_list = spk_src_list
        self.spk_trg_list = spk_trg_list
        self.n_spk_src = len(self.spk_src_list)
        self.n_spk_trg = len(self.spk_trg_list)
        self.n_spk = self.n_spk_src + self.n_spk_trg
        self.n_cyc = n_cyc

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        featfile_src = self.file_list[idx]

        h_src = read_hdf5(featfile_src, "/feat_org_lf0")
        flen_src = h_src.shape[0]
        src_code = np.zeros((flen_src,self.n_spk))

        cv_src_list, src_trg_code_list, featfile_spk, featfile_src_trg, pair_spk_list, src_class_code, trg_class_code_list = proc_multspk_data_random_cls(featfile_src, self.spk_src_list, self.spk_trg_list, self.n_cyc, src_code)

        spcidx_src = read_hdf5(featfile_src, "/spcidx_range")[0]
        flen_spc_src = spcidx_src.shape[0]

        h_src_trg = read_hdf5(featfile_src_trg, "/feat_org_lf0")
        flen_src_trg = h_src_trg.shape[0]
        spcidx_src_trg = read_hdf5(featfile_src_trg, "/spcidx_range")[0]
        flen_spc_src_trg = spcidx_src_trg.shape[0]

        h_src = torch.FloatTensor(self.pad_transform(h_src))
        src_code = torch.FloatTensor(self.pad_transform(src_code))
        src_class_code = torch.LongTensor(self.pad_transform(src_class_code))

        for i in range(self.n_cyc):
            cv_src_list[i] = torch.FloatTensor(self.pad_transform(cv_src_list[i]))
            src_trg_code_list[i] = torch.FloatTensor(self.pad_transform(src_trg_code_list[i]))
            trg_class_code_list[i] = torch.LongTensor(self.pad_transform(trg_class_code_list[i]))

        spcidx_src = torch.LongTensor(self.pad_transform(spcidx_src))

        h_src_trg = torch.FloatTensor(self.pad_transform(h_src_trg))
        spcidx_src_trg = torch.LongTensor(self.pad_transform(spcidx_src_trg))

        return {'h_src': h_src, 'flen_src': flen_src, 'src_code': src_code, 'src_trg_code_list': src_trg_code_list, 'cv_src_list': cv_src_list, 'spcidx_src': spcidx_src, 'flen_spc_src': flen_spc_src, \
                'h_src_trg': h_src_trg, 'flen_src_trg': flen_src_trg, 'spcidx_src_trg': spcidx_src_trg, 'flen_spc_src_trg': flen_spc_src_trg, 'featfile_src': featfile_src, 'featfile_src_trg': featfile_src_trg, \
                'featfile_spk': featfile_spk, 'pair_spk_list': pair_spk_list, 'src_class_code': src_class_code, 'trg_class_code_list': trg_class_code_list}


class FeatureDatasetMultEvalVAECls(Dataset):
    """Dataset for evaluation many-to-many with classifier
    """

    def __init__(self, file_list_src_list, file_list_trg_list, pad_transform, spk_src_list, spk_trg_list):
        self.file_list_src_list = file_list_src_list
        self.file_list_trg_list = file_list_trg_list
        self.pad_transform = pad_transform
        self.spk_src_list = spk_src_list
        self.spk_trg_list = spk_trg_list
        self.n_spk_src = len(self.spk_src_list)
        self.n_spk_trg = len(self.spk_trg_list)
        self.n_spk = self.n_spk_src + self.n_spk_trg
        self.n_eval_utt = len(self.file_list_src_list[0])
        self.file_list_src = []
        self.file_list_src_trg = []
        self.count_spk_pair_cv = {}
        for i in range(self.n_spk_src):
            self.count_spk_pair_cv[self.spk_src_list[i]] = {}
            for j in range(self.n_spk_trg):
                self.count_spk_pair_cv[self.spk_src_list[i]][self.spk_trg_list[j]] = 0
        # deterministically select a conv. pair for each validation utterance
        if self.n_spk_trg > 1:
            idx_even_trg = 1
        else:
            idx_even_trg = 0
        idx_odd_trg = 0
        for spk_src_idx in range(self.n_spk_src):
            if spk_src_idx%2 == 0:
                if idx_even_trg >= self.n_spk_trg:
                    if self.n_spk_trg > 1:
                        idx_even_trg = 1
                    else:
                        idx_even_trg = 0
                spk_trg_idx = idx_even_trg
                idx_even_trg += 2
            else:
                if idx_odd_trg >= self.n_spk_trg:
                    idx_odd_trg = 0
                spk_trg_idx = idx_odd_trg
                idx_odd_trg += 2
            for i in range(self.n_eval_utt):
                self.count_spk_pair_cv[self.spk_src_list[spk_src_idx]][self.spk_trg_list[spk_trg_idx]] += 1
                self.file_list_src.append(self.file_list_src_list[spk_src_idx][i])
                self.file_list_src_trg.append(self.file_list_trg_list[spk_trg_idx][i])

    def __len__(self):
        return len(self.file_list_src)

    def __getitem__(self, idx):
        featfile_src = self.file_list_src[idx]
        featfile_src_trg = self.file_list_src_trg[idx]

        spk_src = os.path.basename(os.path.dirname(featfile_src))
        spk_trg = os.path.basename(os.path.dirname(featfile_src_trg))
        for i in range(self.n_spk_src):
            if spk_src == self.spk_src_list[i]:
                idx_src = i
                break
        for i in range(self.n_spk_trg):
            if spk_trg == self.spk_trg_list[i]:
                idx_trg = self.n_spk_src+i
                break

        h_src = read_hdf5(featfile_src, "/feat_org_lf0")
        flen_src = h_src.shape[0]
        src_code = np.zeros((flen_src,self.n_spk))
        src_trg_code = np.zeros((flen_src,self.n_spk))
        src_code[:,idx_src] = 1
        src_trg_code[:,idx_trg] = 1
        cv_src = read_hdf5(featfile_src, "/cvuvlogf0fil_ap_"+spk_trg)
        spcidx_src = read_hdf5(featfile_src, "/spcidx_range")[0]
        src_class_code = np.ones(h_src.shape[0],dtype=np.int64)*idx_src
        src_trg_class_code = np.ones(h_src.shape[0],dtype=np.int64)*idx_trg
        flen_spc_src = spcidx_src.shape[0]

        h_src_trg = read_hdf5(featfile_src_trg, "/feat_org_lf0")
        flen_src_trg = h_src_trg.shape[0]
        trg_code = np.zeros((flen_src_trg,self.n_spk))
        trg_src_code = np.zeros((flen_src_trg,self.n_spk))
        trg_code[:,idx_trg] = 1
        trg_src_code[:,idx_src] = 1
        cv_trg = read_hdf5(featfile_src_trg, "/cvuvlogf0fil_ap_"+spk_src)
        spcidx_src_trg = read_hdf5(featfile_src_trg, "/spcidx_range")[0]
        trg_class_code = np.ones(h_src_trg.shape[0],dtype=np.int64)*idx_trg
        trg_src_class_code = np.ones(h_src_trg.shape[0],dtype=np.int64)*idx_src
        flen_spc_src_trg = spcidx_src_trg.shape[0]

        h_src = torch.FloatTensor(self.pad_transform(h_src))
        src_code = torch.FloatTensor(self.pad_transform(src_code))
        src_trg_code = torch.FloatTensor(self.pad_transform(src_trg_code))
        cv_src = torch.FloatTensor(self.pad_transform(cv_src))
        spcidx_src = torch.LongTensor(self.pad_transform(spcidx_src))
        src_class_code = torch.LongTensor(self.pad_transform(src_class_code))
        src_trg_class_code = torch.LongTensor(self.pad_transform(src_trg_class_code))

        h_src_trg = torch.FloatTensor(self.pad_transform(h_src_trg))
        trg_code = torch.FloatTensor(self.pad_transform(trg_code))
        trg_src_code = torch.FloatTensor(self.pad_transform(trg_src_code))
        cv_trg = torch.FloatTensor(self.pad_transform(cv_trg))
        spcidx_src_trg = torch.LongTensor(self.pad_transform(spcidx_src_trg))
        trg_class_code = torch.LongTensor(self.pad_transform(trg_class_code))
        trg_src_class_code = torch.LongTensor(self.pad_transform(trg_src_class_code))

        return {'h_src': h_src, 'flen_src': flen_src, 'src_code': src_code, 'src_trg_code': trg_code, 'cv_src': cv_src, 'spcidx_src': spcidx_src, 'flen_spc_src': flen_spc_src, \
                'h_src_trg': h_src_trg, 'flen_src_trg': flen_src_trg, 'trg_code': trg_code, 'trg_src_code': trg_src_code, 'cv_trg': cv_trg, 'spcidx_src_trg': spcidx_src_trg, \
                'flen_spc_src_trg': flen_spc_src_trg, 'featfile_src': featfile_src, 'featfile_src_trg': featfile_src_trg, 'src_class_code': src_class_code, 'src_trg_class_code': src_trg_class_code, \
                'trg_class_code': trg_class_code, 'trg_src_class_code': trg_src_class_code}


def validate_length(x, y, upsampling_factor=0):
    """FUNCTION TO VALIDATE LENGTH

    Args:
        x (ndarray): numpy.ndarray with x.shape[0] = len_x
        y (ndarray): numpy.ndarray with y.shape[0] = len_y
        upsampling_factor (int): upsampling factor

    Returns:
        (ndarray): length adjusted x with same length y
        (ndarray): length adjusted y with same length x
    """
    if upsampling_factor == 0:
        if x.shape[0] < y.shape[0]:
            y = y[:x.shape[0]]
        if x.shape[0] > y.shape[0]:
            x = x[:y.shape[0]]
        assert len(x) == len(y)
    else:
        mod_sample = x.shape[0] % upsampling_factor
        if mod_sample > 0:
            x = x[:-mod_sample]
        if x.shape[0] > y.shape[0] * upsampling_factor:
            x = x[:-(x.shape[0]-y.shape[0]*upsampling_factor)]
        elif x.shape[0] < y.shape[0] * upsampling_factor:
            y = y[:-((y.shape[0]*upsampling_factor-x.shape[0])//upsampling_factor)]
        assert len(x) == len(y) * upsampling_factor

    return x, y


class FeatureDatasetNeuVoco(Dataset):
    """Dataset for neural vocoder
    """

    def __init__(self, wav_list, feat_list, pad_wav_transform, pad_feat_transform, upsampling_factor, string_path, wav_transform=None):
        self.wav_list = wav_list
        self.feat_list = feat_list
        self.pad_wav_transform = pad_wav_transform
        self.pad_feat_transform = pad_feat_transform
        self.upsampling_factor = upsampling_factor
        self.string_path = string_path
        self.wav_transform = wav_transform

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        wavfile = self.wav_list[idx]
        featfile = self.feat_list[idx]
        
        x, _ = sf.read(wavfile, dtype=np.float32)
        h = read_hdf5(featfile, self.string_path)

        x, h = validate_length(x, h, self.upsampling_factor)

        if self.wav_transform is not None:
            x = self.wav_transform(x)
        
        slen = x.shape[0]
        flen = h.shape[0]

        if self.wav_transform is not None:
            x = torch.LongTensor(self.pad_wav_transform(x))
        else:
            x = torch.FloatTensor(self.pad_wav_transform(x))
        h = torch.FloatTensor(self.pad_feat_transform(h)).transpose(0,1)

        return {'x': x, 'h': h, 'slen': slen, 'flen': flen, 'wavfile': wavfile}
