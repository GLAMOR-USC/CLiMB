export WANDB_API_KEY=8cd0c45d6a9418a2997ec6478116a01c14499820
export TOKENIZERS_PARALLELISM=false

task_arr=("imagenet")
nshot_arr=(16 32)
subseed_arr=(10 50 100 500 1000)

for t in ${task_arr[@]}
do
    for n in ${nshot_arr[@]}
    do
        for s in ${subseed_arr[@]}
        do
            echo "n-shot: $n, sample_seed: $s"
            python -m train.train_imagenet --encoder_name vilt \
                                    --pretrained_model_name dandelin/vilt-b32-mlm \
                                    --task_name $t \
                                    --output_dir /data/experiments/MCL/vision_only \
                                    --wandb_project_name vl-cl \
                                    --batch_size 32 \
                                    --model_catog vilt-v-cls \
                                    --num_shot $n \
                                    --subsample_seed $s
        done
    done
done
