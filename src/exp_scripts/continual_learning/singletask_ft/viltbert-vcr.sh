export WANDB_API_KEY=8cd0c45d6a9418a2997ec6478116a01c14499820
export TOKENIZERS_PARALLELISM=false

/home/tejas/anaconda3/envs/mcl/bin/python -m train.train_cl --encoder_name viltbert \
                        --pretrained_model_name dandelin/vilt-b32-mlm \
                        --ordered_cl_tasks vcr \
                        --cl_algorithm singletask_ft \
                        --mcl_data_dir /home/shared/MCL/ \
            		--do_train \
                        --output_dir /home/shared/MCL/experiments/ \
                        --wandb_project_name vl-cl \
                        --batch_size 64
