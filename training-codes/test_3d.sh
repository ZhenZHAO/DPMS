# - - - - - - - -     Testing      - - - - - - - # 
expname="LA_visualization"
version="10"
numlb=4 # 4, 8, 16

python ./code/test_performance_3d.py \
    --root_path ./data/LA/ \
    --res_path ./results/LA/ \
    --gpu 3 \
    --exp ${expname}/v${version} \
    --labeled_num ${numlb}
