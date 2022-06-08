export WANDB_API_KEY=8cd0c45d6a9418a2997ec6478116a01c14499820
export TOKENIZERS_PARALLELISM=false

python -W ignore -m train.train_lowshot_multimodal --encoder_name vilt \
                                --pretrained_model_name dandelin/vilt-b32-mlm \
                                --ordered_cl_tasks nlvr2 \
                                --cl_algorithm singletask_ft \
                                --mcl_data_dir /data/datasets/MCL/ \
                                --output_dir /data/experiments/MCL/ \
                                --wandb_project_name vl-lowshot \
                                --batch_size 64