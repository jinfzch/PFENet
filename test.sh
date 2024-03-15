#!/bin/sh
PARTITION=Segmentation

dataset=pascal # pascal coco
exp_name=split0

exp_dir=exp/${dataset}/${exp_name}/${net}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_${net}.yaml

mkdir -p ${model_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp test.sh test.py ${config} ${exp_dir}

python3 -u test.py --config=${config} 2>&1 | tee ${result_dir}/test-$now.log
