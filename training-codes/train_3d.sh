# - - - - - - - - - - - - - - - - - - - - - - - - #

expname="LA_visualization"

log_dir=./logs/${expname}
mkdir -p ${log_dir}

gpuid=3

version="2"


nohup python ./code/train_post_3d.py \
    --gpu_id=${gpuid} \
    --cfg config_3d.yml \
    --exp ${expname}/v${version} \
    >${log_dir}/log_v${version}.log 2>&1 &
