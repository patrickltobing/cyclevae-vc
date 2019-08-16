#!/bin/bash
##################################################################################################
#    SCRIPT FOR ONE-TO-ONE NON-PARALLEL VOICE CONVERSION WITH VAE/CycleVAE/VQ-VAE/CycleVQ-VAE    #
##################################################################################################

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
# based on PyTorch implementation for WaveNet vocoder by Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# USER SETTINGS {{{
#######################################
#           STAGE SETTING             #
#######################################
# {{{
# 0: data preparation step
# 1: feature extraction step
# a: calculate F0 range and power threshold of speakers
# 2: statistics calculation step
# 3: feature converted extraction
# 4: training step
# 5: GV statistics computation
# 6: decoding step
# }}}
#stage=0
#stage=1
#stage=a
#stage=2
#stage=3
#stage=0123
#stage=4
#stage=5
#stage=6
#stage=56

#######################################
#          FEATURE SETTING            #
#######################################
# {{{
# shiftms: shift length in msec (default=5)
# fftl: fft length (default=1024)
# highpass_cutoff: highpass filter cutoff frequency (if 0, will not apply)
# mcep_dim: dimension of mel-cepstrum
# mcep_alpha: alpha value of mel-cepstrum
# n_jobs: number of parallel jobs
# }}}
shiftms=5
fftl=1024
highpass_cutoff=70
#fs=16000
fs=22050
#fs=24000
#fs=44100
#fs=48000
#mcep_dim=24
#mcep_dim=34
mcep_dim=49
#mcep_alpha=0.41000000000000003 #16k
mcep_alpha=0.455 #22.05k
#mcep_alpha=0.466 #24k
#mcep_alpha=0.544 #44.1k
#mcep_alpha=0.554 #48k
#n_jobs=1
#n_jobs=10
n_jobs=40
#n_jobs=50

#######################################
#          TRAINING SETTING           #
######################################
# {{{
# in_dim: dimension of input features
# lat_dim: dimension of latent features
# out_dim: dimension of output features
# n_cyc: number of cycle for CycleVAE/CycleVQ-VAE, 0 means VAE/VQ-VAE
# hidden_layers: number of hidden layers for GRU
# hidden_units: number of hidden units for GRU
# kernel_size: number of kernel size for aux. convolution
# dilation_size: number of dilation size for aux. convolution
# lr: learning rate
# do_prob: dropout probability
# batch_size: batch frame size
# }}}

spks_src=(VCC2SF1)
#spks_src=(VCC2SM1)
#spks_src=(VCC2SF2)
#spks_src=(VCC2SM2)
spks_trg=(VCC2TF1)
#spks_trg=(VCC2TM1)
#spks_trg=(VCC2TF2)
#spks_trg=(VCC2TM2)

#train_src=tr35_"$(IFS=_; echo "${spks_src[*]}")"_"$(IFS=_; echo "${spks_trg[*]}")"
#train_src_trg=trt35_"$(IFS=_; echo "${spks_src[*]}")"_"$(IFS=_; echo "${spks_trg[*]}")"
#eval_src=ev35_"$(IFS=_; echo "${spks_src[*]}")"_"$(IFS=_; echo "${spks_trg[*]}")"
#test_src=ts35_"$(IFS=_; echo "${spks_src[*]}")"_"$(IFS=_; echo "${spks_trg[*]}")"
#train_trg=tr35_"$(IFS=_; echo "${spks_trg[*]}")"_"$(IFS=_; echo "${spks_src[*]}")"
#train_trg_src=trt35_"$(IFS=_; echo "${spks_trg[*]}")"_"$(IFS=_; echo "${spks_src[*]}")"
#eval_trg=ev35_"$(IFS=_; echo "${spks_trg[*]}")"_"$(IFS=_; echo "${spks_src[*]}")"
#test_trg=ts35_"$(IFS=_; echo "${spks_trg[*]}")"_"$(IFS=_; echo "${spks_src[*]}")"

train_src=tr50_"$(IFS=_; echo "${spks_src[*]}")"_"$(IFS=_; echo "${spks_trg[*]}")"
train_src_trg=trt50_"$(IFS=_; echo "${spks_src[*]}")"_"$(IFS=_; echo "${spks_trg[*]}")"
eval_src=ev50_"$(IFS=_; echo "${spks_src[*]}")"_"$(IFS=_; echo "${spks_trg[*]}")"
test_src=ts50_"$(IFS=_; echo "${spks_src[*]}")"_"$(IFS=_; echo "${spks_trg[*]}")"
train_trg=tr50_"$(IFS=_; echo "${spks_trg[*]}")"_"$(IFS=_; echo "${spks_src[*]}")"
train_trg_src=trt50_"$(IFS=_; echo "${spks_trg[*]}")"_"$(IFS=_; echo "${spks_src[*]}")"
eval_trg=ev50_"$(IFS=_; echo "${spks_trg[*]}")"_"$(IFS=_; echo "${spks_src[*]}")"
test_trg=ts50_"$(IFS=_; echo "${spks_trg[*]}")"_"$(IFS=_; echo "${spks_src[*]}")"

spk_src=${spks_src[0]}
spk_trg=${spks_trg[0]}

echo $spk_src $spk_trg

if [ $fs -eq 22050 ]; then
    stdim=4
elif [ $fs -eq 24000 ]; then
    stdim=5
elif [ $fs -eq 48000 ]; then
    stdim=8
elif [ $fs -eq 44100 ]; then
    stdim=7
else
    stdim=4
fi

#in_dim=28
#in_dim=39
in_dim=54

#out_dim=25
#out_dim=35
out_dim=50

#lat_dim=16
lat_dim=32
#lat_dim=50
#lat_dim=64

#n_cyc=0
#n_cyc=1
n_cyc=2
#n_cyc=3
#n_cyc=4

#n_centroids=16
#n_centroids=32
n_centroids=50
#n_centroids=64

hidden_layers=1

hidden_units=1024

kernel_size=3
dilation_size=2

lr=1e-4

weight_decay=0.0

#batch_size=0
batch_size=80

#batch_size_utt_init=1
#batch_size_utt_init=8

batch_size_utt=1
#batch_size_utt=8

batch_size_utt_eval=1
#batch_size_utt_eval=8

epoch_count=500

do_prob=0.5

mdl_name="cyclevae_gauss"
#mdl_name="cyclevqvae"

echo $mdl_name

GPU_device=0
#GPU_device=1
#GPU_device=2

min_idx=80 #sf1-tf1 batch_utt=1
#min_idx=139 #sf1-tf1 batch_utt=8

n_smpl_dec=300

#echo $min_idx

n_gpus=1

# parse options
. parse_options.sh

# stop when error occured
set -e
# }}}


# STAGE 0 {{{
if [ `echo ${stage} | grep 0` ];then
    echo "###########################################################"
    echo "#                 DATA PREPARATION STEP                   #"
    echo "###########################################################"
    mkdir -p data/${train_src}
    mkdir -p data/${train_trg_src}
    mkdir -p data/${eval_src}
    mkdir -p data/${test_src}
    [ -e data/${train_src}/wav.scp ] && rm data/${train_src}/wav.scp
    [ -e data/${train_trg_src}/wav.scp ] && rm data/${train_trg_src}/wav.scp
    [ -e data/${eval_src}/wav.scp ] && rm data/${eval_src}/wav.scp
    [ -e data/${test_src}/wav.scp ] && rm data/${test_src}/wav.scp
    for spk in ${spks_src[@]};do
        find wav/${spk} -name "*.wav" | sort | head -n 40 >> data/${train_src}/wav.scp
        find wav/${spk} -name "*.wav" | sort | tail -n 41 >> data/${train_trg_src}/wav.scp
        find wav/eval/${spk} -name "*.wav" | sort >> data/${test_src}/wav.scp
    done
    mkdir -p data/${train_trg}
    mkdir -p data/${train_src_trg}
    mkdir -p data/${eval_trg}
    mkdir -p data/${test_trg}
    [ -e data/${train_trg}/wav.scp ] && rm data/${train_trg}/wav.scp
    [ -e data/${train_src_trg}/wav.scp ] && rm data/${train_src_trg}/wav.scp
    [ -e data/${eval_trg}/wav.scp ] && rm data/${eval_trg}/wav.scp
    [ -e data/${test_trg}/wav.scp ] && rm data/${test_trg}/wav.scp
    for spk in ${spks_trg[@]};do
        find wav/${spk} -name "*.wav" | sort | tail -n 41 >> data/${train_trg}/wav.scp
        find wav/${spk} -name "*.wav" | sort | head -n 40 >> data/${train_src_trg}/wav.scp
        find wav/eval/${spk} -name "*.wav" | sort >> data/${test_trg}/wav.scp
    done
fi
# }}}


# STAGE 1 {{{
if [ `echo ${stage} | grep 1` ];then
    echo "###########################################################"
    echo "#               FEATURE EXTRACTION STEP                   #"
    echo "###########################################################"
    if true; then
    #if false; then
        # extract feat and wav_anasyn src_speaker
        nj=0
        for set in ${train_src} ${train_trg_src} ${test_src};do
            echo $set
            expdir=exp/feature_extract/${set}
            mkdir -p $expdir
            for spk in ${spks_src[@]}; do
                echo $spk
                minf0=`cat conf/${spk}.f0 | awk '{print $1}'`
                maxf0=`cat conf/${spk}.f0 | awk '{print $2}'`
                pow=`cat conf/${spk}.pow | awk '{print $1}'`
                echo $minf0 $maxf0 $pow
                scp=${expdir}/wav_${spk}.scp
                n_wavs=`cat data/${set}/wav.scp | grep ${spk} | wc -l`
                echo $n_wavs
                if [ $n_wavs -gt 0 ]; then
                    cat data/${set}/wav.scp | grep ${spk} > ${scp}
                    ${train_cmd} --num-threads ${n_jobs} ${expdir}/feature_extract_${spk}.log \
                        feature_extract_vc.py \
                            --expdir exp/feature_extract \
                            --waveforms ${scp} \
                            --wavdir wav_anasyn/${set}/${spk} \
                            --hdf5dir hdf5/${set}/${spk} \
                            --fs ${fs} \
                            --shiftms ${shiftms} \
                            --minf0 ${minf0} \
                            --maxf0 ${maxf0} \
                            --pow ${pow} \
                            --mcep_dim ${mcep_dim} \
                            --mcep_alpha ${mcep_alpha} \
                            --fftl ${fftl} \
                            --highpass_cutoff ${highpass_cutoff} \
                            --n_jobs ${n_jobs}
        
                    # check the number of feature files
                    n_feats=`find hdf5/${set}/${spk} -name "*.h5" | wc -l`
                    echo "${n_feats}/${n_wavs} files are successfully processed."

                    # update job counts
                    nj=$(( ${nj}+1  ))
                    if [ ! ${max_jobs} -eq -1 ] && [ ${max_jobs} -eq ${nj} ];then
                        wait
                        nj=0
                    fi
                fi
            done
        done
        # extract feat and wav_anasyn trg_speaker
        nj=0
        for set in ${train_trg} ${train_src_trg} ${test_trg};do
            echo $set
            expdir=exp/feature_extract/${set}
            mkdir -p $expdir
            for spk in ${spks_trg[@]}; do
                echo $spk
                minf0=`cat conf/${spk}.f0 | awk '{print $1}'`
                maxf0=`cat conf/${spk}.f0 | awk '{print $2}'`
                pow=`cat conf/${spk}.pow | awk '{print $1}'`
                echo $minf0 $maxf0 $pow
                scp=${expdir}/wav_${spk}.scp
                n_wavs=`cat data/${set}/wav.scp | grep ${spk} | wc -l`
                echo $n_wavs
                if [ $n_wavs -gt 0 ]; then
                    cat data/${set}/wav.scp | grep ${spk} > ${scp}
                    ${train_cmd} --num-threads ${n_jobs} ${expdir}/feature_extract_${spk}.log \
                        feature_extract_vc.py \
                            --expdir exp/feature_extract \
                            --waveforms ${scp} \
                            --wavdir wav_anasyn/${set}/${spk} \
                            --hdf5dir hdf5/${set}/${spk} \
                            --fs ${fs} \
                            --shiftms ${shiftms} \
                            --minf0 ${minf0} \
                            --maxf0 ${maxf0} \
                            --pow ${pow} \
                            --mcep_dim ${mcep_dim} \
                            --mcep_alpha ${mcep_alpha} \
                            --fftl ${fftl} \
                            --highpass_cutoff ${highpass_cutoff} \
                            --n_jobs ${n_jobs}
        
                    # check the number of feature files
                    n_feats=`find hdf5/${set}/${spk} -name "*.h5" | wc -l`
                    echo "${n_feats}/${n_wavs} files are successfully processed."

                    # update job counts
                    nj=$(( ${nj}+1  ))
                    if [ ! ${max_jobs} -eq -1 ] && [ ${max_jobs} -eq ${nj} ];then
                        wait
                        nj=0
                    fi
                fi
            done
        done
    fi
    # make scp for feats
    for set in ${train_src} ${train_trg_src} ${train_trg} ${train_src_trg} ${test_src} ${test_trg};do
        echo $set
        find hdf5/${set} -name "*.h5" | sort > tmp2
        if [ "$set" = ${train_src} ] || [ "$set" = ${train_trg_src} ] || [ "$set" = ${eval_src} ] || [ "$set" = ${test_src} ]; then
            rm -f data/${set}/feats.scp
            for spk in ${spks_src[@]}; do
                cat tmp2 | grep "\/${spk}\/" >> data/${set}/feats.scp
            done
        else
            rm -f data/${set}/feats.scp
            for spk in ${spks_trg[@]}; do
                cat tmp2 | grep "\/${spk}\/" >> data/${set}/feats.scp
            done
        fi
        rm -f tmp2
    done
fi
# }}}


# STAGE a {{{
if [ `echo ${stage} | grep a` ];then
    echo "###########################################################"
    echo "#              INIT SPEAKER STATISTICS STEP               #"
    echo "###########################################################"
    expdir=exp/init_spk_stat
    if true; then
    #if false; then
        for spk in ${spks_src[@]};do
            echo $spk
            cat data/${train_src}/feats.scp | grep \/${spk}\/ > data/${train_src}/feats_spk-${spk}.scp
            ${train_cmd} exp/init_spk_stat/init_stat_${train_src}_spk-${spk}.log \
                spk_stat.py \
                    --expdir ${expdir} \
                    --feats data/${train_src}/feats_spk-${spk}.scp \
                    --spkr ${spk}
        done
        echo "source init spk statistics are successfully calculated."
        for spk in ${spks_trg[@]};do
            echo $spk
            cat data/${train_trg}/feats.scp | grep \/${spk}\/ > data/${train_trg}/feats_spk-${spk}.scp
            ${train_cmd} exp/init_spk_stat/init_stat_${train_trg}_spk-${spk}.log \
                spk_stat.py \
                    --expdir ${expdir} \
                    --feats data/${train_trg}/feats_spk-${spk}.scp \
                    --spkr ${spk}
        done
        echo "target init spk statistics are successfully calculated."
    fi
fi
# }}}


# STAGE 2 {{{
if [ `echo ${stage} | grep 2` ];then
    echo "###########################################################"
    echo "#              CALCULATE STATISTICS STEP                  #"
    echo "###########################################################"
    expdir=exp/calculate_statistics
    if true; then
    #if false; then
        echo ${train_src}
        ${train_cmd} exp/calculate_statistics/calc_stats_${train_src}.log \
            calc_stats_vc.py \
                --expdir ${expdir} \
                --feats data/${train_src}/feats.scp \
                --stats data/${train_src}/stats.h5
        echo "source statistics are successfully calculated."
        echo ${train_trg}
        ${train_cmd} exp/calculate_statistics/calc_stats_${train_trg}.log \
            calc_stats_vc.py \
                --expdir ${expdir} \
                --feats data/${train_trg}/feats.scp \
                --stats data/${train_trg}/stats.h5
        echo "target statistics are successfully calculated."
        ${train_cmd} exp/calculate_statistics/calc_stats_${train_src}_${train_trg}.log \
            calc_stats_vc_joint.py \
                --expdir exp/calculate_statistics \
                --feats_src data/${train_src}/feats.scp \
                --feats_trg data/${train_trg}/feats.scp \
                --stats data/${train_src}/stats_jnt.h5
        echo "joint statistics are successfully calculated."
    fi
fi
# }}}


# STAGE 3 {{{
if [ `echo ${stage} | grep 3` ];then
    echo "######################################################"
    echo "#              EXTRACT CV FEATURES                   #"
    echo "######################################################"
    expdir=exp/feature_extract_cv
    if true; then
    #if false; then
        echo ${train_src} ${train_src_trg}
        ${train_cmd} exp/feature_extract_cv/${train_src}-${train_src_trg}.log \
            feature_cv_extract_vc.py \
                --expdir ${expdir} \
                --fs ${fs} \
                --feats_src data/${train_src}/feats.scp \
                --feats_trg data/${train_src_trg}/feats.scp \
                --stats_src data/${train_src}/stats.h5 \
                --stats_trg data/${train_trg}/stats.h5
        echo "train cv source features are successfully extracted."
        echo ${train_trg} ${train_trg_src}
        ${train_cmd} exp/feature_extract_cv/${train_trg}-${train_trg_src}.log \
            feature_cv_extract_vc.py \
                --expdir ${expdir} \
                --fs ${fs} \
                --feats_src data/${train_trg}/feats.scp \
                --feats_trg data/${train_trg_src}/feats.scp \
                --stats_src data/${train_trg}/stats.h5 \
                --stats_trg data/${train_src}/stats.h5
        echo "train cv target features are successfully extracted."
        echo ${test_src} ${test_trg}
        ${train_cmd} exp/feature_extract_cv/${test_src}-${test_trg}.log \
            feature_cv_extract_vc.py \
                --expdir ${expdir} \
                --fs ${fs} \
                --feats_src data/${test_src}/feats.scp \
                --feats_trg data/${test_trg}/feats.scp \
                --stats_src data/${train_src}/stats.h5 \
                --stats_trg data/${train_trg}/stats.h5
        echo "test cv features are successfully extracted."
    fi
fi
# }}}


# STAGE 4 {{{
# set variables
spk_src_list="$(IFS=_; echo "${spks_src[*]}")"
spk_trg_list="$(IFS=_; echo "${spks_trg[*]}")"

setting=22.05k_${mdl_name}_${spk_src_list}-${spk_trg_list}_hl${hidden_layers}_hu${hidden_units}_ld${lat_dim}_ks${kernel_size}_ds${dilation_size}_cyc${n_cyc}_lr${lr}_bs${batch_size}_wd${weight_decay}_do${do_prob}_epoch${epoch_count}_bsu${batch_size_utt}_bsue${batch_size_utt_eval}
#setting=22.05k_${mdl_name}_${spk_src_list}-${spk_trg_list}_hl${hidden_layers}_hu${hidden_units}_ld${lat_dim}_ks${kernel_size}_ds${dilation_size}_cyc${n_cyc}_lr${lr}_bs${batch_size}_wd${weight_decay}_do${do_prob}_epoch${epoch_count}_nctr${n_centroids}_bsui${batch_size_utt_init}_bsu${batch_size_utt}_bsue${batch_size_utt_eval}

#expdir=exp/tr35_${setting}
expdir=exp/tr50_${setting}
if [ `echo ${stage} | grep 4` ];then
    echo "###############################################"
    echo "#               TRAINING STEP                 #"
    echo "###############################################"
    mkdir -p ${expdir} 
    echo $expdir
    if true; then
    #if false; then
            #train_gru_cyclevqvae_batch.py \
        #${cuda_cmd} ${expdir}/log/${spk_src_list}-${spk_trg_list}_resume-${min_idx}.log \
        ${cuda_cmd} ${expdir}/log/${spk_src_list}-${spk_trg_list}.log \
            train_gru_cyclevae_gauss_batch.py \
                --expdir ${expdir} \
                --feats_src data/${train_src}/feats.scp \
                --feats_src_trg data/${train_src_trg}/feats.scp \
                --feats_trg data/${train_trg}/feats.scp \
                --feats_trg_src data/${train_trg_src}/feats.scp \
                --feats_eval_src data/${test_src}/feats.scp \
                --feats_eval_trg data/${test_trg}/feats.scp \
                --stats_src data/${train_src}/stats.h5 \
                --stats_trg data/${train_trg}/stats.h5 \
                --stats_jnt data/${train_src}/stats_jnt.h5 \
                --spk_src ${spk_src} \
                --spk_trg ${spk_trg} \
                --in_dim ${in_dim} \
                --out_dim ${out_dim} \
                --stdim ${stdim} \
                --lr ${lr} \
                --kernel_size ${kernel_size} \
                --dilation_size ${dilation_size} \
                --epoch_count ${epoch_count} \
                --hidden_units ${hidden_units} \
                --hidden_layers ${hidden_layers} \
                --do_prob ${do_prob} \
                --weight_decay ${weight_decay} \
                --lat_dim ${lat_dim} \
                --n_cyc ${n_cyc} \
                --GPU_device ${GPU_device} \
                --batch_size_utt ${batch_size_utt} \
                --batch_size_utt_eval ${batch_size_utt_eval} \
                --batch_size ${batch_size}
                #--resume ${expdir}/checkpoint-${min_idx}.pkl \
                #--batch_size_utt_init ${batch_size_utt_init} \
                #--n_centroids ${n_centroids} \
    fi
fi
# }}}


# STAGE 5 {{{
if [ `echo ${stage} | grep 5` ];then
    echo ${expdir}
    if true; then
    #if false; then
        echo "##############################################################"
        echo "#                  CONVERTED_GV_COMPUTATION                  #"
        echo "##############################################################"
        #outdir=${expdir}/cvgv_${min_idx}
        outdir=${expdir}/cvgv_${min_idx}-${n_smpl_dec}
        mkdir -p $outdir
        feats=data/${train_src}/feats.scp
        feats_scp=${outdir}/feats_tr_${spk_src}.scp
        cat $feats | grep "\/${spk_src}\/" > ${feats_scp}
        feats_trg=data/${train_src_trg}/feats.scp
        feats_trg_scp=${outdir}/feats_tr_${spk_trg}.scp
        cat $feats_trg | grep "\/${spk_trg}\/" > ${feats_trg_scp}
        config=${expdir}/model.conf
        model=${expdir}/checkpoint-${min_idx}.pkl

        #feats_v=data/${train_trg_src}/feats.scp
        #feats_scp_v=${outdir}/feats_trv_${spk_src}.scp
        #cat $feats_v | grep "\/${spk_src}\/" > ${feats_scp_v}
        #feats_trg_v=data/${train_trg}/feats.scp
        #feats_trg_scp_v=${outdir}/feats_trv_${spk_trg}.scp
        #cat $feats_trg_v | grep "\/${spk_trg}\/" > ${feats_trg_scp_v}
        #cat $feats_scp_v >> $feats_scp
        #cat $feats_trg_scp_v >> $feats_trg_scp
        #rm -f $feats_scp_v $feats_trg_scp_v

        #${cuda_cmd} ${expdir}/log/decode_cvgv_tr_${spk_src}-${spk_trg}_${min_idx}.log \
            #calc_cvgv_gru-cyclevqvae.py \
        ${cuda_cmd} ${expdir}/log/decode_cvgv_tr_${spk_src}-${spk_trg}_${min_idx}-${n_smpl_dec}.log \
            calc_cvgv_gru-cyclevae_gauss.py \
                --feats ${feats_scp} \
                --feats_trg ${feats_trg_scp} \
                --stats_src data/${train_src}/stats.h5 \
                --stats_trg data/${train_trg}/stats.h5 \
                --stats_jnt data/${train_src}/stats_jnt.h5 \
                --outdir ${outdir} \
                --model ${model} \
                --config ${config} \
                --n_gpus ${n_gpus} \
                --GPU_device ${GPU_device} \
                --n_smpl_dec ${n_smpl_dec} \
                --write_gv true
    fi
fi
# }}}


# STAGE 6 {{{
if [ `echo ${stage} | grep 6` ];then
    config=${expdir}/model.conf
    model=${expdir}/checkpoint-${min_idx}.pkl
    #outdir=${expdir}/wav_cv_${spk_src}-${spk_trg}_${min_idx}
    outdir=${expdir}/wav_cv_${spk_src}-${spk_trg}_${min_idx}-${n_smpl_dec}
    minf0=`cat conf/${spk_src}.f0 | awk '{print $1}'`
    maxf0=`cat conf/${spk_src}.f0 | awk '{print $2}'`
    minf0_trg=`cat conf/${spk_trg}.f0 | awk '{print $1}'`
    maxf0_trg=`cat conf/${spk_trg}.f0 | awk '{print $2}'`

    waveforms=data/${test_src}/wav.scp
    waveforms_trg=data/${test_trg}/wav.scp
    feats=data/${test_src}/feats.scp

    pow=`cat conf/${spk_src}.pow | awk '{print $1}'`
    pow_trg=`cat conf/${spk_trg}.pow | awk '{print $1}'`

    if true; then
    #if false; then
        mkdir -p ${outdir}
        echo "######################################################"
        echo "#                  DECODING SPECNET                  #"
        echo "######################################################"
        #${cuda_cmd} ${expdir}/log/decode_ev_${spk_src}-${spk_trg}_${min_idx}.log \
            #decode_gru-cyclevqvae.py \
        ${cuda_cmd} ${expdir}/log/decode_ev_${spk_src}-${spk_trg}_${min_idx}-${n_smpl_dec}.log \
            decode_gru-cyclevae_gauss.py \
                --waveforms ${waveforms} \
                --feats ${feats} \
                --outdir ${outdir} \
                --model ${model} \
                --config ${config} \
                --fs ${fs} \
                --mcep_alpha ${mcep_alpha} \
                --fftl ${fftl} \
                --shiftms ${shiftms} \
                --n_gpus ${n_gpus} \
                --minf0 ${minf0} \
                --maxf0 ${maxf0} \
                --stats_src data/${train_src}/stats.h5 \
                --stats_trg data/${train_trg}/stats.h5 \
                --stats_jnt data/${train_src}/stats_jnt.h5 \
                --waveforms_trg ${waveforms_trg} \
                --minf0_trg ${minf0_trg} \
                --maxf0_trg ${maxf0_trg} \
                --pow ${pow} \
                --pow_trg ${pow_trg} \
                --GPU_device ${GPU_device} \
                --n_smpl_dec ${n_smpl_dec} \
                --intervals 10
    fi
fi
# }}}

