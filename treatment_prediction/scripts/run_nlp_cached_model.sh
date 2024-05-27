#!/bin/bash
trap ctrl_c INT

cd ../nlp/

echo $pwd

init_cmd=$1

output_folder=$2

export CUDA_VISIBLE_DEVICES=$3

treat_opt=$4

full_cached_model_folder=$5

id_ls=()

for ((i=1; i<=3; i++))
#for i in "${id_ls[@]}";
do
	echo "repeat ihdp iteration $i"
	curr_log_folder=$output_folder/$treat_opt
	mkdir -p $curr_log_folder
	curr_log_folder=$curr_log_folder/$i
	mkdir -p $curr_log_folder
#	cmd="python train_tabular_rl.py --epochs 2000 --num_treatments 2 --dataset_name ihdp --batch_size 256 --lr 1e-3 --topk_act 1 --program_max_len 4 --log_folder /data6/wuyinjun/causal_tabular/ihdp/logs/ --model_config configs/configs_ihdp.yaml --method ours --dataset_id $i"
	cmd="$init_cmd --log_folder $curr_log_folder --treatment_opt ${treat_opt} --seed ${i} --backbone TransTEE --regression_ratio 5 --log_folder $curr_log_folder --cached_backbone ${full_cached_model_folder}/${i}_bestmod.pt"
	echo $cmd
        $cmd > $output_folder/output_$i.txt 2>&1

done
