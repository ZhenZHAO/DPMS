# - - - - - - - - - - - - - - - - - - - - - - - - #

expname="ACDC_abls_recent"
log_dir=./logs/${expname}
mkdir -p ${log_dir}


gpuid=3

version="3"


nohup python ./code/train_post.py \
    --gpu_id=${gpuid} \
    --cfg config_2d.yml \
    --exp ${expname}/v${version} \
    >${log_dir}/log_v${version}.log 2>&1 &

