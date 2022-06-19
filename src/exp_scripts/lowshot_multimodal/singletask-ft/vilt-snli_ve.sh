export TOKENIZERS_PARALLELISM=false

python -m train.train_lowshot_multimodal --encoder_name vilt \
                        --pretrained_model_name dandelin/vilt-b32-mlm \
                        --ordered_cl_tasks snli-ve \
                        --cl_algorithm singletask_ft \
                        --climb_data_dir /data/datasets/MCL/ \
                        --output_dir /data/experiments/MCL/ \
                        --batch_size 64
