export WANDB_API_KEY=8cd0c45d6a9418a2997ec6478116a01c14499820
export TOKENIZERS_PARALLELISM=false

task_arr=("sst2" "imdb")
nshot_arr=(16 32)
subseed_arr=(10 50 100)
ckpt_arr=(
    "/data/experiments/MCL/viltbert-experience_replay-task0_vqa-task1_nlvr2-task2_snli-ve-task3_vcr/checkpoints/task1_nlvr2/encoder" \
    "/data/experiments/MCL/viltbert-experience_replay-task0_vqa-task1_nlvr2-task2_snli-ve-task3_vcr/checkpoints/task2_snli-ve/encoder" \
    "/data/experiments/MCL/viltbert-experience_replay-task0_vqa-task1_nlvr2-task2_snli-ve-task3_vcr/checkpoints/task3_vcr/encoder" \
    )

for t in ${task_arr[@]}
do
    for s in ${subseed_arr[@]}
    do
        for n in ${nshot_arr[@]}
        do
            for c in ${ckpt_arr[@]}
            do
                echo "ckpt: $c, n-shot: $n, sample_seed: $s"
                python -m train.train_language --encoder_name vilt \
                                        --pretrained_model_name $c \
                                        --task_name $t \
                                        --output_dir /data/experiments/MCL/lang_only/viltbert \
                                        --wandb_project_name vl-cl \
                                        --batch_size 16 \
                                        --model_catog vilt-l-seq \
                                        --num_shot $n \
                                        --subsample_seed $s
            done
        done
    done
done
