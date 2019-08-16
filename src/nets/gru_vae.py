# -*- coding: utf-8 -*-

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
# based on PyTorch implementation for WaveNet vocoder by Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import logging
import sys
import time
import math

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np


def initialize(m):
    """FUNCTION TO INITILIZE NETWORK PARAMETERS

    Arg:
        m (torch.nn.Module): torch nn module instance
    """
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)
        else:
            logging.info("ERROR: " + name)


class TwoSidedDilConv1d(nn.Module):
    """1D TWO-SIDED DILATED CONVOLUTION"""

    def __init__(self, in_dim=39, kernel_size=3, layers=2):
        super(TwoSidedDilConv1d, self).__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.layers = layers
        self.rec_field = self.kernel_size**self.layers
        self.padding = int((self.rec_field-1)/2)
        self.conv = nn.ModuleList()
        for i in range(self.layers):
            if i > 0:
                self.conv += [nn.Conv1d(self.in_dim*(self.kernel_size**(i)), self.in_dim*(self.kernel_size**(i+1)), self.kernel_size, stride=1, dilation=self.kernel_size**i, padding=0)]
            else:
                self.conv += [nn.Conv1d(self.in_dim, self.in_dim*(self.kernel_size**(i+1)), self.kernel_size, stride=1, dilation=1, padding=self.padding)]

    def forward(self, x):
        """Forward calculation

        Arg:
            x (Variable): float tensor variable with the shape  (B x C x T)

        Return:
            (Variable): float tensor variable with the shape (B x C x T)
        """
        x = self.conv[0](x)
        for i in range(1,self.layers):
            x = self.conv[i](x)

        return x


def sampling_vae(param, lat_dim=None, training=False, relu_vae=False):
    if lat_dim is None:
        lat_dim = int(param.shape[1]/2)
    mu = param[:,:lat_dim]
    sigma = param[:,lat_dim:]
    if training:
        eps = torch.randn(param.shape[0], lat_dim).cuda()
    else:
        with torch.no_grad():
            eps = torch.randn(param.shape[0], lat_dim).cuda()
    if not relu_vae:
        return mu + torch.exp(sigma/2) * eps # log_var
    else:
        return mu + torch.sqrt(sigma) * eps # var


def sampling_vae_batch(param, lat_dim=None, training=False, relu_vae=False):
    if lat_dim is None:
        lat_dim = int(param.shape[2]/2)
    mu = param[:,:,:lat_dim]
    sigma = param[:,:,lat_dim:]
    if training:
        eps = torch.randn(param.shape[0], param.shape[1], lat_dim).cuda()
    else:
        with torch.no_grad():
            eps = torch.randn(param.shape[0], param.shape[1], lat_dim).cuda()
    if not relu_vae:
        return mu + torch.exp(sigma/2) * eps # log_var
    else:
        return mu + torch.sqrt(sigma) * eps # var


def sampling_vae_laplace(param, lat_dim=None, training=False, relu_vae=False):
    if lat_dim is None:
        lat_dim = int(param.shape[1]/2)
    mu = param[:,:lat_dim]
    sigma = param[:,lat_dim:]
    if training:
        eps = torch.empty(param.shape[0], lat_dim).cuda().uniform_(-0.4999,0.5)
    else:
        with torch.no_grad():
            eps = torch.empty(param.shape[0], lat_dim).cuda().uniform_(-0.4999,0.5)
    if not relu_vae:
        return mu - torch.exp(sigma) * eps.sign() * torch.log1p(-2*eps.abs()) # log_scale
    else:
        return mu - sigma * eps.sign() * torch.log1p(-2*eps.abs()) # scale


def loss_vae(param, lat_dim=None, relu_vae=False):
    if lat_dim is None:
        lat_dim = int(param.shape[1]/2)
    mu = param[:,:lat_dim]
    sigma = param[:,lat_dim:]
    if not relu_vae:
        return torch.mean(0.5*torch.sum(torch.exp(sigma)+torch.pow(mu,2)-sigma-1.0,1)) # log_var
        #return torch.sum(0.5*torch.sum(torch.exp(sigma)+torch.pow(mu,2)-sigma-1.0,1)) # log_var
    else:
        return torch.mean(0.5*torch.sum(sigma+torch.pow(mu,2)-torch.log(sigma)-1.0,1)) # var
        #return torch.sum(0.5*torch.sum(sigma+torch.pow(mu,2)-torch.log(sigma)-1.0,1)) # var


def loss_vae_laplace(param, lat_dim=None, relu_vae=False):
    if lat_dim is None:
        lat_dim = int(param.shape[1]/2)
    mu = param[:,:lat_dim]
    sigma = param[:,lat_dim:]
    mu_abs = mu.abs()
    if not relu_vae:
        scale = torch.exp(sigma)
        return torch.mean(torch.sum(-sigma+scale*torch.exp(-mu_abs/scale)+mu_abs-1,1)) # log_scale
        #return torch.sum(torch.sum(-sigma+scale*torch.exp(-mu_abs/scale)+mu_abs-1,1)) # log_scale
    else:
        return torch.mean(torch.sum(-torch.log(sigma)+sigma*torch.exp(-mu_abs/sigma)+mu_abs-1,1)) # scale
        #return torch.sum(torch.sum(-torch.log(sigma)+sigma*torch.exp(-mu_abs/sigma)+mu_abs-1,1)) # scale

        #torch.mean(-log_scale+torch.exp(log_scale)*torch.exp(-median.abs()/torch.exp(log_scale))+median.abs()-1)


def nn_search(encoding, centroids):
    #similarity = torch.matmul(encoding,centroids.transpose(0,1)) # T x K
    #z2 = torch.sum(centroids**2, -1) # K
    #x2 = torch.sum(encoding**2, -1) # T
    #dist2 = torch.add(similarity,z2) + x2.unsqueeze(-1)
    T = encoding.shape[0]
    K = centroids.shape[0]
    #dist2 = torch.sum((encoding.unsqueeze(1).repeat(1,K,1)-centroids.unsqueeze(0).repeat(T,1,1)).pow(2),2) # T x K
    dist2 = torch.sum((encoding.unsqueeze(1).repeat(1,K,1)-centroids.unsqueeze(0).repeat(T,1,1)).abs(),2) # T x K
    #logging.info(dist2)
    ctr_ids = torch.argmin(dist2, dim=-1)

    return ctr_ids


def nn_search_batch(encoding, centroids):
    #similarity = torch.matmul(encoding,centroids.transpose(0,1)) # T x K
    #z2 = torch.sum(centroids**2, -1) # K
    #x2 = torch.sum(encoding**2, -1) # T
    #dist2 = torch.add(similarity,z2) + x2.unsqueeze(-1)
    B = encoding.shape[0]
    T = encoding.shape[1]
    K = centroids.shape[0]
    #dist2 = torch.sum((encoding.unsqueeze(1).repeat(1,K,1)-centroids.unsqueeze(0).repeat(T,1,1)).pow(2),2) # T x K
    dist2 = torch.sum((encoding.unsqueeze(2).repeat(1,1,K,1)-centroids.unsqueeze(0).unsqueeze(0).repeat(B,T,1,1)).abs(),3) # B x T x K
    #logging.info(dist2)
    ctr_ids = torch.argmin(dist2, dim=-1) # B x T

    return ctr_ids


def weighted_ctr(encoding, centroids):
    #similarity = torch.matmul(encoding,centroids.transpose(0,1)) # T x K
    #z2 = torch.sum(centroids**2, -1) # K
    #x2 = torch.sum(encoding**2, -1) # T
    #dist2 = torch.add(similarity,z2) + x2.unsqueeze(-1)
    T = encoding.shape[0]
    K = centroids.shape[0]
    D = centroids.shape[1]
    #dist2 = torch.sum((encoding.unsqueeze(1).repeat(1,K,1)-centroids.unsqueeze(0).repeat(T,1,1))**2,1) # T x K
    centroids_tiled = centroids.unsqueeze(0).repeat(T,1,1)
    dist = torch.sum((encoding.unsqueeze(1).repeat(1,K,1)-centroids_tiled).abs(),2) # T x K
    score = torch.exp(-dist) # T x K
    post = score/torch.sum(score,1).unsqueeze(1).repeat(1,K) # T x K
    weighted_centroids = torch.sum(post.unsqueeze(2).repeat(1,1,D)*centroids_tiled,1) # T x D
    weighted_dist = torch.sum(dist*post,1).mean()
    return weighted_centroids, weighted_dist
    #logging.info(dist2)
    #ctr_ids = torch.argmin(dist2, dim=-1)

    #return ctr_ids


class GMM(nn.Module):
    def __init__(self, n_mix=64, n_dim=16):
        super(GMM, self).__init__()
        self.n_mix = n_mix
        self.n_dim = n_dim
        self.pi2 = 6.283185307
        self.c = 1.0/np.sqrt(np.power(self.pi2,self.n_dim))
        self.wghts = nn.Embedding(self.n_mix, 1)
        self.means = nn.Embedding(self.n_mix, self.n_dim)
        self.dcovs = nn.Embedding(self.n_mix, self.n_dim)

    def forward(self, data):
        T = data.shape[0]
        #logging.info(torch.prod(self.dcovs.weight,1))
        c_dets = (1.0/torch.sqrt(torch.prod(self.dcovs.weight,1))).unsqueeze(0).repeat(T,1) # T x K
        means = self.means.weight.unsqueeze(0).repeat(T,1,1) # T x K x C
        dprecs = (1.0/self.dcovs.weight).unsqueeze(0).repeat(T,1,1) # T x K x C
        mhbs_dist = torch.sum(dprecs*(data.unsqueeze(1).repeat(1,self.n_mix,1)-means)**2,2) # T x K
        exp = torch.exp(-0.5*mhbs_dist) # T x K
        probs = self.wghts.weight.unsqueeze(0).repeat(T,1) * self.c * c_dets * exp # T x K
        likes = torch.sum(probs,1) # T
        log_likes = torch.mean(torch.log(likes)) # 1
        posteriors = probs/likes.unsqueeze(1).repeat(1,self.n_mix) # T x K
        #logging.info(posteriors)
        e_means = torch.sum(posteriors.unsqueeze(2).repeat(1,1,self.n_dim)*means,1) # T x C
        #logging.info(e_means)
    
        return log_likes, e_means

    def probs(self, data):
        T = data.shape[0]
        c_dets = (1.0/torch.sqrt(torch.prod(self.dcovs.weight,1))).unsqueeze(0).repeat(T,1) # T x K
        dprecs = (1.0/self.dcovs.weight).unsqueeze(0).repeat(T,1,1) # T x K x C
        mhbs_dist = torch.sum(dprecs*(data.unsqueeze(1).repeat(1,self.n_mix,1)-self.means.weight.unsqueeze(0).repeat(T,1,1))**2,2) # T x K
        exp = torch.exp(-0.5*mhbs_dist) # T x K
        probs = self.wghts.weight.unsqueeze(0).repeat(T,1) * self.c * c_dets * exp # T x K
        likes = torch.sum(probs,1) # T
        log_likes = torch.mean(torch.log(likes)) # 1
    
        return log_likes

    def update(self, data):
        T = data.shape[0]
        c_dets = (1.0/torch.sqrt(torch.prod(self.dcovs.weight,1))).unsqueeze(0).repeat(T,1) # T x K
        means = self.means.weight.unsqueeze(0).repeat(T,1,1) # T x K x C
        dprecs = (1.0/self.dcovs.weight).unsqueeze(0).repeat(T,1,1) # T x K x C
        data = data.unsqueeze(1).repeat(1,self.n_mix,1) # T x K x C
        mhbs_dist = torch.sum(dprecs*(data-means)**2,2) # T x K
        exp = torch.exp(-0.5*mhbs_dist) # T x K
        probs = self.wghts.weight.unsqueeze(0).repeat(T,1) * self.c * c_dets * exp # T x K
        likes = torch.sum(probs,1) # T
        log_likes = torch.mean(torch.log(likes)) # 1
        posteriors = probs/likes.unsqueeze(1).repeat(1,self.n_mix) # T x K
        wghts = torch.mean(posteriors,0) # K
        sum_posteriors = torch.sum(posteriors,0).unsqueeze(1).repeat(1,self.n_dim) # K x C
        posteriors = posteriors.unsqueeze(2).repeat(1,1,self.n_dim) # T x K x C
        means = torch.sum(posteriors*data,0)/sum_posteriors # K x C
        dcovs = torch.clamp(torch.sum(posteriors*(data-means.unsqueeze(0).repeat(T,1,1))**2,0)/sum_posteriors,min=1e-6) # K x C
        self.wghts.weight = nn.Parameter(wghts)
        self.means.weight = nn.Parameter(means)
        self.dcovs.weight = nn.Parameter(dcovs)
    
        return log_likes


class GRU_RNN(nn.Module):
    """GRU-RNN for FEATURE MAPPING

    Args:
        in_dim (int): input dimension
        out_dim (int): RNN output dimension
        hidden_units (int): GRU hidden units amount
        hidden_layers (int): GRU hidden layers amount
        kernel_size (int): kernel size for input convolutional layers
        dilation_size (int): dilation size for input convolutional layers
        do_prob (float): drop-out probability
        scale_in_flag (bool): flag to use input normalization layer
        scale_out_flag (bool): flag to use output un-normalization layer
        scale_in_out_flag (bool): flag to use output normalization layer for after performing input normalization (use this if you are injecting Gaussian noise)
        [For normalization / un-normalization layers, you need to set the weights & biases manually according to training data statistics, and DON'T include them in parameters optimization]
    """

    def __init__(self, in_dim=39, out_dim=35, hidden_units=1024, hidden_layers=1, kernel_size=3, dilation_size=2, do_prob=0, scale_in_flag=True, scale_out_flag=True, scale_in_out_flag=False):
        super(GRU_RNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.do_prob = do_prob
        self.scale_in_flag = scale_in_flag
        self.scale_out_flag = scale_out_flag
        self.scale_in_out_flag = scale_in_out_flag

        # Normalization layer
        if self.scale_in_flag:
            self.scale_in = nn.Conv1d(self.in_dim, self.in_dim, 1)

        # Dilated two-sides-causality convolution layers: +- 13 frames with kernel_size = 3 and dilation_size = 3
        self.conv = TwoSidedDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, layers=self.dilation_size)
        self.receptive_field = self.conv.rec_field
        self.tot_in_dim = self.in_dim*self.receptive_field+self.out_dim
        if self.do_prob > 0:
            self.conv_drop = nn.Dropout(p=self.do_prob)

        # GRU layer(s)
        if self.do_prob > 0 and self.hidden_layers > 1:
            self.gru = nn.GRU(self.tot_in_dim, self.hidden_units, self.hidden_layers, dropout=self.do_prob, batch_first=True)
        else:
            self.gru = nn.GRU(self.tot_in_dim, self.hidden_units, self.hidden_layers, batch_first=True)

        if self.do_prob > 0:
            self.gru_drop = nn.Dropout(p=self.do_prob)
        self.out_1 = nn.Conv1d(self.hidden_units, self.out_dim, 1)

        # Un-normalization layer
        if self.scale_out_flag:
            self.scale_out = nn.Conv1d(self.out_dim, self.out_dim, 1)
        if self.scale_in_out_flag:
            self.scale_in_out = nn.Conv1d(self.out_dim, self.out_dim, 1)

    def forward(self, x, y_in, softmax=False, sigmoid=False, exp=False, h_in=None, noise=0, res=False, res_stdim=0, res_endim=35, do=False, clamp_vae=False, relu_vae=False, lat_dim=16, clamp_vae_laplace=False):
        """Forward calculation

        Args:
            x (Variable): float tensor variable with the shape  (T x C_in)

        Return:
            (Variable): float tensor variable with the shape (T x C_out)
        """
        if len(x.shape) > 2:
            batch_flag = True
            T = x.shape[1]
            # Normalization layer
            if self.scale_in_flag:
                x_in = self.scale_in(x.transpose(1,2)) # B x T x C -> B x C x T
            else:
                x_in = x.transpose(1,2) # B x T x C -> B x C x T
        else:
            batch_flag = False
            T = x.shape[0]
            # Normalization layer
            if self.scale_in_flag:
                x_in = self.scale_in(torch.unsqueeze(x.transpose(0,1),0)) # T x C -> C x T -> B x C x T
            else:
                x_in = torch.unsqueeze(x.transpose(0,1),0) # T x C -> C x T -> B x C x T

        if noise > 0:
            #logging.info('noise: %lf' % (noise))
            x_noise = torch.normal(mean=0, std=noise*torch.ones(x_in.shape[0],x_in.shape[1],x_in.shape[2])).cuda()
            x_in = x_in + x_noise # B x C x T

        # Dilated two-sides-causality convolution layers: +- 13 frames with kernel_size = 3 and dilation_size = 3
        if self.do_prob > 0 and do:
            x_conv = self.conv_drop(self.conv(x_in).transpose(1,2)) # T x C --> B x C x T --> B x T x C
        else:
            x_conv = self.conv(x_in).transpose(1,2) # T x C --> B x C x T --> B x T x C

        if res or self.scale_in_out_flag:
            x_in = x_in.transpose(1,2) # B x T x C

        # GRU and AR layers
        # 1st frame
        if h_in is None:
            out, h = self.gru(torch.cat((x_conv[:,:1],y_in),2)) # B x T x C
        else:
            out, h = self.gru(torch.cat((x_conv[:,:1],y_in),2), h_in) # B x T x C
        if self.do_prob > 0 and do:
            out = self.gru_drop(out)
        if not res:
            y_in = self.out_1(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
        else:
            y_in = x_in[:,:1,res_stdim:res_endim] + self.out_1(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
        trj = y_in
        # 2nd-Tth frame
        if self.do_prob > 0 and do:
            if not res:
                for i in range(1,T):
                    out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                    out = self.gru_drop(out)
                    y_in = self.out_1(out.transpose(1,2)).transpose(1,2)
                    trj = torch.cat((trj, y_in), 1)
            else:
                for i in range(1,T):
                    out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                    out = self.gru_drop(out)
                    y_in = x_in[:,i:(i+1),res_stdim:res_endim] + self.out_1(out.transpose(1,2)).transpose(1,2)
                    trj = torch.cat((trj, y_in), 1)
        else:
            if not res:
                for i in range(1,T):
                    out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                    y_in = self.out_1(out.transpose(1,2)).transpose(1,2)
                    trj = torch.cat((trj, y_in), 1)
            else:
                for i in range(1,T):
                    out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                    y_in = x_in[:,i:(i+1),res_stdim:res_endim] + self.out_1(out.transpose(1,2)).transpose(1,2)
                    trj = torch.cat((trj, y_in), 1)

        # Un-normalization layer
        if self.scale_out_flag:
            if batch_flag:
                trj_out = self.scale_out(trj.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
            else:
                trj_out = torch.squeeze(self.scale_out(trj.transpose(1,2)).transpose(1,2),0) # B x T x C -> B x C x T -> T x C
        else:
            if batch_flag:
                trj_out = trj
                if clamp_vae:
                    if not relu_vae:
                        trj_out = torch.cat((trj_out[:,:,:lat_dim],torch.clamp(trj_out[:,:,lat_dim:],min=-13.815510557964274104107948728106)),2)
                    else:
                        trj_out = torch.cat((trj_out[:,:,:lat_dim],torch.clamp(F.relu(trj_out[:,:,lat_dim:]),min=1e-6)),2)
                elif clamp_vae_laplace:
                    if not relu_vae:
                        trj_out = torch.cat((trj_out[:,:,:lat_dim],torch.clamp(trj_out[:,:,lat_dim:],min=-7.2543288692621097067625904247823)),2)
                    else:
                        trj_out = torch.cat((trj_out[:,:,:lat_dim],torch.clamp(F.relu(trj_out[:,:,lat_dim:]),min=1e-6)),2)
                elif relu_vae:
                    trj_out = torch.cat((trj_out[:,:,:lat_dim],F.relu(trj_out[:,:,lat_dim:])),2)
            else:
                trj_out = trj.view(-1,self.out_dim)
                if clamp_vae:
                    if not relu_vae:
                        trj_out = torch.cat((trj_out[:,:lat_dim],torch.clamp(trj_out[:,lat_dim:],min=-13.815510557964274104107948728106)),1)
                    else:
                        trj_out = torch.cat((trj_out[:,:lat_dim],torch.clamp(F.relu(trj_out[:,lat_dim:]),min=1e-6)),1)
                elif clamp_vae_laplace:
                    if not relu_vae:
                        trj_out = torch.cat((trj_out[:,:lat_dim],torch.clamp(trj_out[:,lat_dim:],min=-7.2543288692621097067625904247823)),1)
                    else:
                        trj_out = torch.cat((trj_out[:,:lat_dim],torch.clamp(F.relu(trj_out[:,lat_dim:]),min=1e-6)),1)
                elif relu_vae:
                    trj_out = torch.cat((trj_out[:,:lat_dim],F.relu(trj_out[:,lat_dim:])),1)

        if self.scale_in_out_flag:
            if batch_flag:
                x_in_out = self.scale_in_out(x_in.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                trj = x_in_out + trj_out
            else:
                x_in_out = torch.squeeze(self.scale_in_out(x_in.transpose(1,2)).transpose(1,2),0) # B x T x C -> B x C x T -> T x C
                trj = x_in_out + trj_out

        if exp:
            return (torch.exp(trj_out)-1)/10000, y_in, h
        elif softmax:
            return F.softmax(trj_out, dim=-1), y_in, h
        elif sigmoid:
            return torch.sigmoid(trj_out), y_in, h

        if not self.scale_in_out_flag:
            return trj_out, y_in, h
        else:
            return trj, x_in_out, trj_out, y_in, h

    #@staticmethod
    #def remove_weightnorm(model):
    #    gru_rnn = model
    #    for i in range(gru_rnn.conv.layers):
    #        gru_rnn.conv.conv[i] = nn.utils.remove_weight_norm(gru_rnn.conv.conv[i])
    #    gru_rnn.out_1 = nn.utils.remove_weight_norm(gru_rnn.out_1)
    #    return gru_rnn


class TWFSEloss(nn.Module):
	def __init__(self):
		super(TWFSEloss, self).__init__()
		self.criterion = nn.MSELoss(reduction='none')

	def forward(self, x, y, twf=None, GV=True, rmse=False, L2=True):
		if twf is not None:
			if rmse:
				if L2:
					rmse = torch.sqrt(torch.mean(self.criterion(torch.index_select(x,0,twf), y),0))
				else:
					rmse = torch.mean(torch.abs(y-torch.index_select(x,0,twf)),0)
				out_diff = torch.index_select(x-torch.mean(x,0),0,twf)
				trg_diff = y - torch.mean(y,0)
				corr = torch.sum(out_diff*trg_diff,0)/(torch.sqrt(torch.sum(out_diff*out_diff,0))*torch.sqrt(torch.sum(trg_diff*trg_diff,0)))
				#return rmse, corr
				return torch.mean(rmse), torch.mean(corr)
			else:
				#logging.info(torch.index_select(x,0,twf).shape)
				#logging.info(y.shape)
				#logging.info(x.shape)
				#logging.info(twf)
				#logging.info(torch.index_select(x,0,twf))
				#logging.info(y)
				#logging.info(torch.abs(torch.index_select(x,0,twf)-y))
				#logging.info(torch.sum(torch.abs(torch.index_select(x,0,twf)-y)))
				if L2:
					mcd = (10.0/2.3025850929940456840179914546844)*torch.sqrt(2.0*torch.sum(self.criterion(torch.index_select(x,0,twf), y),1))
				else:
					mcd = (10.0/2.3025850929940456840179914546844)*1.4142135623730950488016887242097*torch.sum(torch.abs(torch.index_select(x,0,twf)-y),1)
					#mcd = torch.abs(torch.index_select(x,0,twf)-y)
				#logging.info(mcd)
				mcd_sum = torch.sum(mcd)
				mcd_mean = torch.mean(mcd)
				mcd_std = torch.std(mcd)
				if GV:
				    #if L2:
					#    #return mcd_sum, mcd_mean, mcd_std, torch.mean(torch.sqrt(self.criterion(torch.log(torch.var(x,0)),torch.log(torch.var(y,0)))))
					#    return mcd_sum, mcd_mean, mcd_std, torch.mean(torch.sqrt(self.criterion(torch.log(torch.var(torch.index_select(x,0,twf),0)),torch.log(torch.var(y,0)))))
				    #else:
					#    #return mcd_sum, mcd_mean, mcd_std, torch.mean(torch.abs(torch.log(torch.var(x,0))-torch.log(torch.var(y,0))))
					#    return mcd_sum, mcd_mean, mcd_std, torch.mean(torch.abs(torch.log(torch.var(torch.index_select(x,0,twf),0))-torch.log(torch.var(y,0))))
					return mcd_sum, mcd_mean, mcd_std, torch.mean(torch.sqrt(self.criterion(torch.log(torch.var(torch.index_select(x,0,twf),0)),torch.log(torch.var(y,0)))))
				return mcd_sum, mcd_mean, mcd_std
		else:
			if rmse:
				if L2:
					rmse = torch.sqrt(torch.mean(self.criterion(x, y),0))
				else:
					rmse = torch.mean(torch.abs(x-y),0)
				out_diff = x - torch.mean(x,0)
				trg_diff = y - torch.mean(y,0)
				corr = torch.sum(out_diff*trg_diff,0)/(torch.sqrt(torch.sum(out_diff*out_diff,0))*torch.sqrt(torch.sum(trg_diff*trg_diff,0)))
				#return rmse, corr
				return torch.mean(rmse), torch.mean(corr)
			else:
				if L2:
					mcd = (10.0/2.3025850929940456840179914546844)*torch.sqrt(2.0*torch.sum(self.criterion(x, y),1))
				else:
					mcd = (10.0/2.3025850929940456840179914546844)*1.4142135623730950488016887242097*torch.sum(torch.abs(x-y),1)
				mcd_sum = torch.sum(mcd)
				mcd_mean = torch.mean(mcd)
				mcd_std = torch.std(mcd)
				if GV:
				    if L2:
					    return mcd_sum, mcd_mean, mcd_std, torch.mean(torch.sqrt(self.criterion(torch.log(torch.var(x,0)),torch.log(torch.var(y,0)))))
				    else:
					    return mcd_sum, mcd_mean, mcd_std, torch.mean(torch.abs(torch.log(torch.var(x,0))-torch.log(torch.var(y,0))))
				return mcd_sum, mcd_mean, mcd_std
