#!/bin/bash
path_to_imagenet="/brain-data/all_datasets/imagenet/ILSVRC/Data/CLS-LOC/"
model="pvig_s_224_gelu"
batch_size=128
pretrain_path="./pvig_ti_78.5.pth.tar"

python train.py ${path_to_imagenet} \
       --model ${model} \
       -b ${batch_size} \
       --pretrain_path ${pretrain_path} \
       --evaluate