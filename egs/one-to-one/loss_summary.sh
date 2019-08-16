#!/bin/sh

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


echo cycvae_laplace_50mcep_32lat_2cyc_sf1-tf1_batch1
awk -f proc_loss_log_vae.awk exp/tr50_22.05k_cyclevae_gauss_VCC2SF1-VCC2TF1_hl1_hu1024_ld32_ks3_ds2_cyc2_lr1e-4_bs80_wd0.0_do0.5_epoch500_bsu1_bsue1/log/VCC2SF1-VCC2TF1.log > cycvae_gauss_50mcep_32lat_2cyc_sf1-tf1_batch1.txt
#awk -f proc_loss_log_vae_resume.awk exp/tr50_22.05k_cyclevae_gauss_VCC2SF1-VCC2TF1_hl1_hu1024_ld32_ks3_ds2_cyc2_lr1e-4_bs80_wd0.0_do0.5_epoch500_bsu1_bsue1/train.log > cycvae_gauss_50mcep_32lat_2cyc_sf1-tf1_batch1.txt

echo cycvae_laplace_50mcep_32lat_2cyc_sf1-tf1_batch8
awk -f proc_loss_log_vae.awk exp/tr50_22.05k_cyclevae_gauss_VCC2SF1-VCC2TF1_hl1_hu1024_ld32_ks3_ds2_cyc2_lr1e-4_bs80_wd0.0_do0.5_epoch500_bsu8_bsue8/log/VCC2SF1-VCC2TF1.log > cycvae_gauss_50mcep_32lat_2cyc_sf1-tf1_batch8.txt
#awk -f proc_loss_log_vae_resume.awk exp/tr50_22.05k_cyclevae_gauss_VCC2SF1-VCC2TF1_hl1_hu1024_ld32_ks3_ds2_cyc2_lr1e-4_bs80_wd0.0_do0.5_epoch500_bsu8_bsue8/train.log > cycvae_gauss_50mcep_32lat_2cyc_sf1-tf1_batch8.txt

