#!/bin/bash
trap ctrl_c INT

cd ../tabular/

echo $pwd

init_cmd=$1

output_folder=$2

export CUDA_VISIBLE_DEVICES=$3

full_cached_model_folder=$4

id_ls=(8 12 25 20 17 24 15 23 26 16 14  3 10 19 18 28 21 22 11  2  7  0  4
  6  1 13  9  5)

for ((i=1; i<=50; i++))
#for i in "${id_ls[@]}";
do
	echo "repeat ihdp iteration $i"
	curr_log_folder=$output_folder/$i
	mkdir -p $curr_log_folder
#	cmd="python train_tabular_rl.py --epochs 2000 --num_treatments 2 --dataset_name ihdp --batch_size 256 --lr 1e-3 --topk_act 1 --program_max_len 4 --log_folder /data6/wuyinjun/causal_tabular/ihdp/logs/ --model_config configs/configs_ihdp.yaml --method ours --dataset_id $i"
	cmd="$init_cmd $i --backbone TransTEE --regression_ratio 5 --log_folder $curr_log_folder --cached_backbone ${full_cached_model_folder}/$i/bestmod.pt --fix_backbone"
	echo $cmd
        $cmd > $output_folder/output_logs_$i.txt 2>&1

done
