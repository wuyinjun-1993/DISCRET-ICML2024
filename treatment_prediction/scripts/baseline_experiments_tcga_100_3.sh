#!/bin/bash
trap ctrl_c INT


#rand_seed_ls=(0 100 200)
rand_seed_ls=(200)


out_root_folder="/data2/wuyinjun/causal_tabular/tcga/"


data_root_folder="/data2/wuyinjun/causal_tabular/"

#method_ls=("rf" "dragonnet" "drnet" "tarnet" "vcnet" "ENRL" "Ganite" "bart" "causal_rf" "dt" "lr" "tvae" "TransTEE" "nam")
#method_ls=("rf" "lr" "dt" "bart")
method_ls=("rf" "dt" "causal_rf")


gpu_id_ls=(3 3 2 2 2 2 2 1 1 1 1 3 3 3)

method_num=${#method_ls[@]}
echo "method_num::"$method_num

last_method_idx=$((method_num - 1))


cd ../tabular/


#for ratio in "${missing_ratio_ls[@]}"; 
#do
	for seed in ${rand_seed_ls[@]};
	do
		for i in "${!method_ls[@]}"; 
		do
			method=${method_ls[$i]}
			gpu_id=${gpu_id_ls[$i]}

#			out_folder=${out_root_folder}ihdp_missing_${ratio}
#	                mkdir -p ${out_folder}
                	out_folder=${out_root_folder}/seed_${seed}
        	        mkdir -p ${out_folder}
	                out_folder=${out_folder}/${method}
			mkdir -p ${out_folder}
			out_folder=${out_folder}/logs100/
			mkdir -p $out_folder
			export CUDA_VISIBLE_DEVICES=$gpu_id
			
			echo "method::$method, random_seed:${seed}"
			echo "output folder:: ${out_folder}"
			cmd="python train_tabular.py --epochs 2000 --num_treatments 3 --dataset_name tcga --batch_size 32 --lr 1e-4 --topk_act 1 --program_max_len 100 --model_config configs/configs_tcga.yaml --data_folder ${data_root_folder}  --method ${method} --seed ${seed}  --log_folder $out_folder"
			#cmd="bash run_ihdp_2.sh "${init_cmd}" "$out_folder" "${gpu_id}
			out_log_file=${out_folder}/output.txt
			echo $cmd
			echo $out_log_file
			echo "$i ${last_method_idx}"
#<<COMMENT
			if [ "$i" -eq "$last_method_idx" ]; then
#				echo "stop"
				${cmd}  > $out_log_file 2>&1
			elif [ "$i" -eq 0 ]; then
#				echo "stop"
                                ${cmd}  > $out_log_file 2>&1 &
				sleep 60
			else
			#	echo "here"
				${cmd}  > $out_log_file 2>&1 &
			fi
#COMMENT
		done

	done

#done

