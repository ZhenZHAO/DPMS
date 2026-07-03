expname="Pancrease_init"

log_dir=./logs/${expname}
mkdir -p ${log_dir}

gpuid=3

version="15"


nohup python ./code/train_post_3d.py \
    --gpu_id=${gpuid} \
    --cfg config_3d_pan.yml \
    --patch_size 96 96 96 \
    --exp ${expname}/v${version} \
    >${log_dir}/log_v${version}.log 2>&1 &
