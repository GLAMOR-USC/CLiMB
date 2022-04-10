export WANDB_API_KEY=8cd0c45d6a9418a2997ec6478116a01c14499820
export TOKENIZERS_PARALLELISM=false

nshot_arr=(32 128 512)
subseed_arr=(10 50 100 500 1000)

for n in ${nshot_arr[@]}
do
    for s in ${subseed_arr[@]}
    do
        echo "n-shot: $n, sample_seed: $s"
        python -m train.train_cl --encoder_name vilt \
                                --pretrained_model_name dandelin/vilt-b32-mlm \
                                --ordered_cl_tasks imdb \
                                --cl_algorithm sequential_ft \
                                --do_train \
                                --output_dir /data/experiments/MCL/ \
                                --wandb_project_name vl-cl \
                                --batch_size 32 \
                                --num_shot $n \
                                --subsample_seed $s
    done
done
