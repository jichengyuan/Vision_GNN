#!/bin/bash
num_gpus=1
gpu_ratio=$(echo "$num_gpus / 8" | bc -l) # 8 is the default number of gpus per node
echo "num_gpus: $num_gpus, default_gpus number is 8, so the gpu_ratio: $gpu_ratio"
path_to_imagenet="/data/all_datasets/imagenette2-320/"
model="pvig_ti_224_gelu"
batch_size=128
scheduler="cosine"
epochs=10 # default 300
opt="adamw"
opt_eps=1e-8
j=8
warmup_lr=$(echo "0.000001* $gpu_ratio" | bc -l)
warmup_epochs=20
repeated_aug=true
remode='pixel'
reprob=0.25
amp=true
weight_decay=5e-2
lr=$(echo "0.002 * $gpu_ratio" | bc -l)
drop=0
drop_path=0.1
mixup=0.8
cutmix=1.0
model_ema=true
model_ema_decay=0.99996
aa='rand-m9-mstd0.5-inc1'
color_jitter=0.4
output_path="outputs"

python train.py ${path_to_imagenet} \
       --model ${model} \
       --sched ${scheduler} \
       --epochs ${epochs} \
       --opt ${opt} \
       -j ${j} \
       --warmup-lr ${warmup_lr} \
       --mixup ${mixup} \
       --cutmix ${cutmix} \
       --model-ema \
       --model-ema-decay ${model_ema_decay} \
       --aa ${aa} \
       --color-jitter ${color_jitter} \
       --warmup-epochs ${warmup_epochs} \
       --opt-eps ${opt_eps} \
       --repeated-aug \
       --remode ${remode} \
       --reprob ${reprob} \
       --reprob ${reprob} \
       --amp \
       --lr ${lr} \
       --weight-decay ${weight_decay} \
       --drop ${drop} \
       --drop-path ${drop_path} \
       -b ${batch_size} \
       --output ${output_path}