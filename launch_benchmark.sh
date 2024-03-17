#!/bin/bash
set -xe

# gen-efficientnet-pytorch
function main {
    # prepare workload
    workload_dir="${PWD}"
    # set common info
    source oob-common/common.sh
    init_params $@
    fetch_device_info
    set_environment
    pip install -r ${workload_dir}/requirements.txt
    cd RCAN_TestCode/code/
    
    # requirements
    if [ "${DATASET_DIR}" == "" ];then
        set +x
        echo "[ERROR] Please set DATASET_DIR before launch"
        echo "  export DATASET_DIR=/path/to/dataset"
        exit 1
        set -x
    fi

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        if [ $batch_size -le 0 ];then
            batch_size=32
        fi
        # pre-run
         python main.py --data_test MyImage --scale 3 --model RCAN \
            --test_only --chop --self_ensemble \
            --n_resgroups 10 --n_resblocks 20 --n_feats 64 \
            --pre_train ${DATASET_DIR}/models_ECCV2018RCAN/RCAN_BIX3.pt \
            --testpath ../LR/LRBI/ --testset Set5 --n_threads 0 --cpu --batch_size $batch_size \
            --num_iter 1 --num_warmup 0 \
            --channels_last ${channels_last} --precision=${precision} --device $device \
            ${addtion_options}
        #
        for batch_size in ${batch_size_list[@]}
        do
            if [ $batch_size -le 0 ];then
                batch_size=32
            fi
            # clean workspace
            logs_path_clean
            # generate launch script for multiple instance
            if [ "${OOB_USE_LAUNCHER}" == "1" ] && [ "${device}" != "cuda" ];then
                generate_core_launcher
            else
                generate_core
            fi
            # launch
            echo -e "\n\n\n\n Running..."
            cat ${excute_cmd_file} |column -t > ${excute_cmd_file}.tmp
            mv ${excute_cmd_file}.tmp ${excute_cmd_file}
            source ${excute_cmd_file}
            echo -e "Finished.\n\n\n\n"
            # collect launch result
            collect_perf_logs
        done
    done
}

function generate_core {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${cpu_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        # instances
        if [ "${device}" != "cuda" ];then
            OOB_EXEC_HEADER=" numactl -m $(echo ${device_array[i]} |awk -F ';' '{print $2}') "
            OOB_EXEC_HEADER+=" -C $(echo ${device_array[i]} |awk -F ';' '{print $1}') "
        else
            OOB_EXEC_HEADER=" CUDA_VISIBLE_DEVICES=${device_array[i]} "
        fi

        printf " ${OOB_EXEC_HEADER} \
            python main.py --data_test MyImage --scale 3 --model RCAN \
                --test_only --chop --self_ensemble \
                --n_resgroups 10 --n_resblocks 20 --n_feats 64 \
                --pre_train ${DATASET_DIR}/models_ECCV2018RCAN/RCAN_BIX3.pt \
                --testpath ../LR/LRBI/ --testset Set5 --n_threads 0 --cpu \
                --channels_last ${channels_last} \
                --num_iter ${num_iter} --num_warmup ${num_warmup} --device $device --batch_size $batch_size \
                --precision=${precision} \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
        if [ "${numa_nodes_use}" == "0" ];then
            break
        fi
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

# run
function generate_core_launcher {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${cpu_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        printf "python -m oob-common.launch --enable_jemalloc \
                    --core_list $(echo ${device_array[@]} |sed 's/;.//g') \
                    --log_file_prefix rcpi${real_cores_per_instance} \
                    --log_path ${log_dir} \
                    --ninstances ${#cpu_array[@]} \
                    --ncore_per_instance ${real_cores_per_instance} \
            main.py --data_test MyImage --scale 3 --model RCAN \
                --test_only --chop --self_ensemble \
                --n_resgroups 10 --n_resblocks 20 --n_feats 64 \
                --pre_train ${DATASET_DIR}/models_ECCV2018RCAN/RCAN_BIX3.pt \
                --testpath ../LR/LRBI/ --testset Set5 --n_threads 0 --cpu \
                --channels_last ${channels_last} \
                --num_iter ${num_iter} --num_warmup ${num_warmup} --device $device --batch_size $batch_size \
                --precision=${precision} \
                ${addtion_options} \
        > /dev/null 2>&1 &  \n" |tee -a ${excute_cmd_file}
        break
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

# download common files
rm -rf oob-common && git clone https://github.com/intel-sandbox/oob-common.git

# Start
main "$@"
