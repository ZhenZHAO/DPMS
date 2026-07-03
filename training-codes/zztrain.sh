
nvidia-smi

# # - - - - - - - - - - - 1. Together training - - - - - - - - - - - - - #
# expname_2d="ACDC_latest"
# log_dir_2d=./logs/${expname_2d}
# mkdir -p ${log_dir_2d}

# expname_3d="LA_latest"
# log_dir_3d=./logs/${expname_3d}
# mkdir -p ${log_dir_3d}


# gpuid=3
# version="8"

# nohup python3.6 ./code/train_post.py \
#     --gpu_id=${gpuid} \
#     --cfg config_2d.yml \
#     --exp ${expname_2d}/v${version} \
#     >${log_dir_2d}/log_v${version}.log 2>&1 &


# nohup python3.6 ./code/train_post_3d.py \
#     --gpu_id=${gpuid} \
#     --cfg config_3d.yml \
#     --exp ${expname_3d}/v${version} \
#     >${log_dir_3d}/log_v${version}.log 2>&1 &


# # # - - - - - - - - - - - 2. 2D training - - - - - - - - - - - - - #
# expname="ACDC_leave"
# log_dir=./logs/${expname}
# mkdir -p ${log_dir}

# gpuid=1
# version="22"
# # cfgfile=config_2d_base.yml # config_2d_base, config_2d, config_2d_complete
# cfgfile=config_2d_complete.yml



# nohup python3.6 ./code/train_post.py \
#     --gpu_id=${gpuid} \
#     --cfg ${cfgfile} \
#     --exp ${expname}/v${version} \
#     >${log_dir}/log_v${version}.log 2>&1 &


# # - - - - - - - - - - - 3. 3D training - - - - - - - - - - - - - #

expname="LA_leave"

log_dir=./logs/${expname}
mkdir -p ${log_dir}

gpuid=0
version="24"

nohup python3.6 ./code/train_post_3d.py \
    --gpu_id=${gpuid} \
    --cfg config_3d.yml \
    --exp ${expname}/v${version} \
    >${log_dir}/log_v${version}.log 2>&1 &

