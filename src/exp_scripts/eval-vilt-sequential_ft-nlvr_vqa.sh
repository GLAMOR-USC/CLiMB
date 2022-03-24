export WANDB_API_KEY=8cd0c45d6a9418a2997ec6478116a01c14499820
export TOKENIZERS_PARALLELISM=false

python -m train.train_cl --encoder_name vilt \
                        --pretrained_model_name dandelin/vilt-b32-mlm \
                        --ordered_cl_tasks nlvr2,vqa \
                        --cl_algorithm sequential_ft \
			--do_eval \
                        --output_dir /data/experiments/MCL/ \
                        --wandb_project_name vl-cl \
                        --batch_size 32