# CLiMB: The Continual Learning in Multimodality Benchmark

CLiMB is a benchmark to study the challenge of learning multimodal tasks in a CL setting, and to systematically evaluate how upstream continual learning can rapidly generalize to new multimodal and unimodal tasks.

[Pre-print](https://tejas1995.github.io/files/climb-arxiv.pdf)

![CLiMB Learning Setting](https://tejas1995.github.io/files/MMCL.jpg)

CLiMB evaluates candidate CL models and learning algorithms in two phases. For Phase I, Upstream Continual Learning, a pre-trained multimodal model is trained on a sequence of vision-and-language tasks, and evaluated after each task on its degree of Forgetting of past tasks and Knowledge Transfer to the next task. For Phase II, after each multimodal task the model is evaluated for its Downstream Low-Shot Transfer capability on both multimodal and unimodal tasks.

## Setup

1. Create Conda environment with Python 3.6

```
conda create -n climb python=3.6
conda activate climb
```

2. Install requirements

```
git clone https://github.com/GLAMOR-USC/CLiMB.git --recurse-submodules
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
cd src/adapter-transformers
pip install -e .
```

## Data

### [Downloading and Organizing Existing Tasks](DATA_DOWNLOAD.md)



### Adding New Tasks

Coming soon!

## Models

### Existing Models

The initial implementation of CLiMB includes two Vision-Language encoders: ViLT and ViLT-BERT. [ViLT](https://arxiv.org/abs/2102.03334) is a Vision-Language Transformer that operates over lanugage inputs and image patches. ViLT-BERT is a modification of ViLT, where the Transformer's language input embeddings are replaced with language representations extracted from a pre-trained frozen BERT.

### Adding New Models

Coming soon!

## Training in CLiMB

### Upstream Continual Learning

Continual Learning experiments can be run using the file `src/train/train_upstream_continual_learning.py`. The following arguments need to be specified:

-   `--encoder_name` : Name of base vision-language encoder to use. Available VL encoders can be viewed [here](src/configs/model_configs.py#L4).
-   `--pretrained_model_name` : Name of pre-trained model checkpoint name to load for encoder.
-   `--ordered_cl_tasks` : Order of CL tasks. Tasks available for upstream CL can be seen [here](src/configs/task_configs.py#L6).
-   `--cl_algorithm` : Name of CL algorithm to use. Some algorithms may include additional algorithm-specific arguments
-   `--climb_data_dir` : Directory where the training data for all CLiMB tasks is located
-   `--do_train` and `--do_eval` (latter if only doing Knowledge Transfer/Forgetting evaluation of an already CL-trained model)
-   `--output_dir` : Directory where experiment outputs will be stored.
-   `--batch_size`

Sample CL training scripts for training a ViLT encoder on the task order VQAv2 -> NLVR2 -> SNLI-VE -> VCR, using a variety of CL algorithms, can be viewed [here](src/exp_scripts/continual_learning/vqa_nlvr_snlive_vcr/).

### Downstream Low-Shot Transfer

#### Low-Shot Multimodal Transfer

TODO: Tejas

#### Low-Shot Language-Only Tasks

TODO: Charlotte

#### Low-Shot Vision-Only Tasks

TODO: Charlotte

## License

MIT License

## Contact

Questions or issues? Contact tejas.srinivasan@usc.edu
