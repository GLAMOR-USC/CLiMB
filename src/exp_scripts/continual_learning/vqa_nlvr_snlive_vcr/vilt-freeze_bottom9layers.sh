export TOKENIZERS_PARALLELISM=false

python -m train.train_upstream_continual_learning --encoder_name vilt \
                        --pretrained_model_name dandelin/vilt-b32-mlm \
                        --ordered_cl_tasks vqa,nlvr2,snli-ve,vcr \
                        --cl_algorithm freeze_bottom_k_layers \
			--layers_to_freeze 9 \
                        --climb_data_dir /home/shared/MCL/ \
                        --do_train \
			--do_eval \
                        --output_dir /home/shared/MCL/experiments/ \
                        --do_wandb_logging \
                        --batch_size 64
