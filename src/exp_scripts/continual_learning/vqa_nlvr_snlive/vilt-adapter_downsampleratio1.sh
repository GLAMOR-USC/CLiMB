export TOKENIZERS_PARALLELISM=false

python -m train.train_upstream_continual_learning --encoder_name vilt \
                        --pretrained_model_name dandelin/vilt-b32-mlm \
                        --ordered_cl_tasks vqa,nlvr2,snli-ve \
                        --cl_algorithm adapter \
			--adapter_config houlsby \
			--adapter_reduction_factor 1 \
			--climb_data_dir /data/datasets/MCL/ \
                        --do_train \
			--do_eval \
                        --output_dir /data/experiments/MCL/ \
                        --do_wandb_logging \
                        --batch_size 64
