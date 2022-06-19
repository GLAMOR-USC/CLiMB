export WANDB_API_KEY=8cd0c45d6a9418a2997ec6478116a01c14499820
export TOKENIZERS_PARALLELISM=false

python -m train.train_upstream_continual_learning --encoder_name vilt \
                        --pretrained_model_name dandelin/vilt-b32-mlm \
                        --ordered_cl_tasks nlvr2,vqa,vcr,snli-ve \
                        --cl_algorithm sequential_ft \
                        --mcl_data_dir /data/datasets/MCL/ \
                        --do_train \
			--do_eval \
                        --output_dir /data/experiments/MCL/ \
                        --wandb_project_name vl-cl \
                        --batch_size 64
