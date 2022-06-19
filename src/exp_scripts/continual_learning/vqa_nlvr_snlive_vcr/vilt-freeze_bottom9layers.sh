export WANDB_API_KEY=8cd0c45d6a9418a2997ec6478116a01c14499820
export TOKENIZERS_PARALLELISM=false

/home/tejas/anaconda3/envs/mcl/bin/python -m train.train_upstream_continual_learning --encoder_name vilt \
                        --pretrained_model_name dandelin/vilt-b32-mlm \
                        --ordered_cl_tasks vqa,nlvr2,snli-ve,vcr \
                        --cl_algorithm freeze_bottom_k_layers \
			--layers_to_freeze 9 \
                        --mcl_data_dir /home/shared/MCL/ \
                        --do_train \
			--do_eval \
                        --output_dir /home/shared/MCL/experiments/ \
                        --wandb_project_name vl-cl \
                        --batch_size 64
