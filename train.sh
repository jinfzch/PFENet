#!/bin/sh
PARTITION=Segmentation

dataset=pascal # pascal coco
exp_name=split0

net=resnet50 # vgg resnet50 mobilenetv3

exp_dir=exp/${dataset}/${exp_name}/${net}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_${net}.yaml

mkdir -p ${model_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp train.sh train.py ${config} ${exp_dir}

python3 -u train.py --config=${config} 2>&1 | tee ${result_dir}/train-$now.log
