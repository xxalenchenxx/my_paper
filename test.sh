#!/bin/bash

# 定義資料集名稱
datasets=(
    # "u_u_643_2280_web-polblogs.mtx"
    # "u_u_1024_3056_delaunay-n10.mtx"
    # "u_u_4096_12264_delaunay-n12.mtx"
    # "u_u_32768_98274_delaunay-n15.mtx"
    # "loc-Gowalla.mtx"
    # "amazon0302-OK.mtx"
    "soc-flickr.mtx"
    "road-roadNet-CA.mtx"
    
    # synthesis
    # "u_100_800.mtx"
    # "u_500_4000.mtx"
    # "u_1000_8000.mtx"
    # "u_2000_16000.mtx"
    # "u_4000_32000.mtx"
    # "u_6000_48000.mtx"
    # "u_8000_64000.mtx"

    # synthesis_avg_5
    # "u_100_250.mtx"
    # "u_500_1250.mtx"
    # "u_1000_2500.mtx"
    # "u_2000_5000.mtx"
    # "u_4000_10000.mtx"
    # "u_6000_15000.mtx"
    # "u_8000_20000.mtx"
    # "u_10000_25000.mtx"
    # "u_15000_37500.mtx"
    # "u_20000_50000.mtx"

    # synthesis_high_MAX_avg_5
    # "u_100_250.mtx"
    # "u_500_1250.mtx"
    # "u_1000_2500.mtx"
    # "u_2000_5000.mtx"
    # "u_4000_10000.mtx"
    # "u_6000_15000.mtx"
    # "u_8000_20000.mtx"
    
)

# 定義參數集
parameters=(32)
# parameters=(32)
# 程式名稱
program="./a"

# 資料集目錄
dataset_dir="../dataset"
# dataset_dir="../dataset/synthesis"
# dataset_dir="../dataset/synthesis_avg_5"
# dataset_dir="../dataset/synthesis_high_MAX_avg_5"
# dataset_dir="../dataset/synthesis_high_MAX_avg_16"

# 遍歷每個資料集和參數組合
for dataset in "${datasets[@]}"; do
    for param in "${parameters[@]}"; do
        echo "正在執行: $program $dataset_dir/$dataset $param"
        $program "$dataset_dir/$dataset" $param
        echo "完成: $dataset with parameter $param"
    done
done

echo "所有資料集執行完成！"
