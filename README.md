# PyTorch Implementation of Non-Parallel Voice Conversion with CycleVAE


----
## Usage
    $cd tools
    $make
    $cd ../egs/one-to-one

open run.sh

set stage=0123 for full feature extraction

    $bash run.sh

*to compute speaker configs, run with stage=1, then with stage=a, then change accordingly, then run stage=1 again*

*computed f0 and power histograms will be stored in exp/init\_spk\_stat*

set stage=4 for training

    $bash run.sh


----
## Stage details
STAGE 0: data list preparation

STAGE 1: feature extraction

STAGE a: calculation of f0 and power threshold statistics for feature extraction [speaker configs are in conf/]

STAGE 2: calculation of feature statistics for model development

STAGE 3: extraction of converted excitation features for cyclic flow

STAGE 4: model training

STAGE 5: calculation of GV statistics of converted mcep

STAGE 6: decoding and waveform conversion


----
## Trained examples

Example of trained models, converted wavs, and logs can be accessed in [trained_example](http://bit.ly/309zWXc)
which used speakers SF1 and TF1 from Voice Conversion Challenge (VCC) 2018.

    $cd cyclevae-vc_trained/egs/one-to-one/

open run.sh

set stage=5 for GV stat calc.

    $bash run.sh

set stage=6 for decoding and wav conversion

    $bash run.sh

one of the example of model, converted wavs and logs are located in exp/tr50\_22.05k\_cyclevae\_gauss_VCC2SF1-VCC2TF1\_hl1\_hu1024\_ld32\_ks3\_ds2\_cyc2\_lr1e-4\_bs80\_wd0.0\_do0.5\_epoch500\_bsu1\_bsue1/

to summarize training log, use

    $sh loss_summary.sh


----
## Soon to be added features
* CycleVQVAE
* Many-to-Many VC with CycleVAE
* Many-to-Many VC with CycleVQVAE

*which have been implemented, will be added after finishing the journal*


----
## Contact
If there are any questions or problems, especially about hyperparameters and other settings, please let me know.

Patrick Lumban Tobing (Patrick)

patrick.lumbantobing@g.sp.m.is.nagoya-u.ac.jp


----
## Reference
P. L. Tobing, Y.-C. Wu, T. Hayashi, K. Kobayashi, and T. Toda, “Non-parallel voice conversion with cyclic
variational autoencoder”, CoRR arXiv preprint arXiv: 1907.10185, 2019. (Accepted for INTERSPEECH 2019)

