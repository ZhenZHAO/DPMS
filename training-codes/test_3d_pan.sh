# - - - - - - - -     Testing      - - - - - - - # 

expname="Pancrease_init"
# version="7"
# numlb=12 # 6, 12
version="15"
numlb=12 # 6, 12

python ./code/test_performance_3d.py \
    --root_path ./data/Pancreas/ \
    --res_path ./results/Pancreas/ \
    --dataset "Pancreas" \
    --gpu 2 \
    --exp ${expname}/v${version} \
    --labeled_num ${numlb}
