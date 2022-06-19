export TOKENIZERS_PARALLELISM=false

python -W ignore -m train.train_upstream_continual_learning --encoder_name vilt \
                                --pretrained_model_name dandelin/vilt-b32-mlm \
                                --ordered_cl_tasks nlvr2 \
                                --cl_algorithm singletask_ft \
                                --climb_data_dir /data/datasets/MCL/ \
                                --do_train \
				--output_dir /data/experiments/MCL/ \
                                --do_wandb_logging \
                                --batch_size 64
