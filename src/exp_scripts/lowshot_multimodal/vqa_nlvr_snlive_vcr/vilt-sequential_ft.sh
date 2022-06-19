export TOKENIZERS_PARALLELISM=false

python -m train.train_lowshot_multimodal --encoder_name vilt \
                        --pretrained_model_name dandelin/vilt-b32-mlm \
                        --ordered_cl_tasks vqa,nlvr2,snli-ve,vcr \
                        --cl_algorithm sequential_ft \
                        --climb_data_dir /data/datasets/MCL/ \
                        --output_dir /data/experiments/MCL/ \
                        --batch_size 64
