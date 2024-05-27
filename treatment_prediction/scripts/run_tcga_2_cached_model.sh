#!/bin/bash
trap ctrl_c INT


rand_seed_ls=(0 100 200)
#rand_seed_ls=(0)


out_root_folder="/data3/wuyinjun/causal_tabular/tcga/"

#method_ls=("rf" "dragonnet" "drnet" "tarnet" "vcnet" "ENRL" "Ganite" "bart" "causal_rf" "dt" "lr" "tvae" "TransTEE" "nam")
#method_ls=("rf" "Ganite" "bart" "causal_rf" "dt" "lr" "tvae")

#gpu_id_ls=(3 3 2 2 2 2 2 1 1 1 1 3 3 3)

#method_num=${#method_ls[@]}
#echo "method_num::"$method_num

#last_method_idx=$((method_num - 1))

topk_act=$1
program_max_len=$2
regress_ratio=$3
suffix=$4
gpu_id=$5
fix_model=$6

cd ../tabular/


#for ratio in "${missing_ratio_ls[@]}"; 
#do
	for seed in ${rand_seed_ls[@]};
	do
#		for i in "${!method_ls[@]}"; 
#		do
			method="ours"
			#${method_ls[$i]}
	#		gpu_id=2 #${gpu_id_ls[$i]}

#			out_folder=${out_root_folder}ihdp_missing_${ratio}
#	                mkdir -p ${out_folder}
                	out_folder=${out_root_folder}/seed_${seed}
        	        mkdir -p ${out_folder}
	                out_folder=${out_folder}/${method}
			mkdir -p ${out_folder}
			out_folder=${out_folder}/logs${suffix}/
			mkdir -p $out_folder
			export CUDA_VISIBLE_DEVICES=$gpu_id
			
			echo "method::$method, random_seed:${seed}"
			echo "output folder:: ${out_folder}"
			cmd="python train_tabular_rl.py --gpu_db  --epochs 500 --num_treatments 3 --dataset_name tcga --batch_size 128 --lr 1e-4 --topk_act ${topk_act} --program_max_len ${program_max_len} --model_config configs/configs_tcga.yaml --data_folder /data3/wuyinjun/causal_tabular/  --method ${method} --seed ${seed}  --log_folder $out_folder --backbone TransTEE --cached_backbone /data3/wuyinjun/causal_tabular/tcga//seed_${seed}/TransTEE/logs/bestmod.pt $fix_model --regression_ratio  ${regress_ratio}"
			#cmd="bash run_ihdp_2.sh "${init_cmd}" "$out_folder" "${gpu_id}
			out_log_file=${out_folder}/output.txt
			echo $cmd
			echo $out_log_file
#<<COMMEN
			#	echo "here"
			${cmd} > $out_log_file 2>&1 
#COMMENT
#		done

	done

#done

