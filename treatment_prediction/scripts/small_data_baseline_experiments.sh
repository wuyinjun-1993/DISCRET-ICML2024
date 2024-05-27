#!/bin/bash
trap ctrl_c INT

#missing_ratio_ls=(0.5 0.4 0.3 0.2 0.1)
missing_ratio_ls=(400 200 100 50 20)  #  (50 30 20)

rand_seed_ls=(100)
#rand_seed_ls=(0)


out_root_folder="/data2/wuyinjun/causal_tabular/"

#method_ls=("rf" "dragonnet" "drnet" "tarnet" "vcnet" "ENRL" "Ganite" "bart" "causal_rf" "dt" "lr" "tvae" "TransTEE" "nam")
#method_ls=("drnet" "tarnet" "vcnet" "lr")
method_ls=("TransTEE" "drnet" "tarnet" "vcnet")

gpu_id_ls=(3 3 2 2 2 2 2 1 1 1 1 3 3 3)

method_num=${#method_ls[@]}
echo "method_num::"$method_num

last_method_idx=$((method_num - 1))

for ratio in "${missing_ratio_ls[@]}"; 
do
	for seed in ${rand_seed_ls[@]};
	do
		for i in "${!method_ls[@]}"; 
		do
			method=${method_ls[$i]}
			gpu_id=${gpu_id_ls[$i]}

			out_folder=${out_root_folder}ihdp_small_data_${ratio}
	                mkdir -p ${out_folder}
                	out_folder=${out_folder}/seed_${seed}
        	        mkdir -p ${out_folder}
	                out_folder=${out_folder}/${method}
			mkdir -p ${out_folder}
			out_folder=${out_folder}/logs/
			mkdir -p $out_folder
			
			echo "method::$method, small_data::${ratio}, random_seed:${seed}"
			echo "output folder:: ${out_folder}"
			init_cmd="python train_tabular.py --epochs 2000 --num_treatments 2 --dataset_name ihdp --batch_size 32 --lr 2e-4 --topk_act 1 --program_max_len 6 --model_config configs/configs_ihdp.yaml --data_folder /data2/wuyinjun/causal_tabular/ --subset_num ${ratio}  --method ${method} --seed ${seed}  --dataset_id"
			cmd="bash run_ihdp_2.sh "${init_cmd}" "$out_folder" "${gpu_id}
			out_log_file=${out_folder}/output_script.txt
			echo $cmd
			echo $out_log_file
			echo "$i ${last_method_idx}"
			if [ "$i" -eq "$last_method_idx" ]; then
#				echo "stop"
				bash run_ihdp_2.sh "${init_cmd}" "$out_folder" ${gpu_id} > $out_log_file 2>&1
			elif [ "$i" -eq 0 ]; then
#				echo "stop"
                                bash run_ihdp_2.sh "${init_cmd}" "$out_folder" ${gpu_id} > $out_log_file 2>&1 &
				sleep 60
			else
			#	echo "here"
				bash run_ihdp_2.sh "${init_cmd}" "$out_folder" ${gpu_id} > $out_log_file 2>&1 &
			fi
		
		done

	done

done
