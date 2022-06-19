# Downloading and Organizing Existing Tasks

## Multimodal Tasks

CLiMB initially includes four vision-and-language tasks: 
- [Visual Question Answering](https://visualqa.org/)  (VQAv2)
- [Natural Language for Visual Reasoning](https://lil.nlp.cornell.edu/nlvr/) (NLVR2)
- [SNLI-Visual Entailment](https://github.com/necla-ml/SNLI-VE) (SNLI-VE) 
- [Visual Commonsense Reasoning](https://visualcommonsense.com/) (VCR)

Data files for these four tasks can be downloaded from their respective websites. The data files are organized as follow:

```
data
├── flickr30k
│   └── flickr30k_images/
├── ms-coco
│   ├── images/
├── nlvr2
│   ├── data
│   │   ├── balanced
│   │   ├── dev.json
│   │   ├── filter_data.py
│   │   ├── test1.json
│   │   ├── train.json
│   │   └── unbalanced
│   └── images
│       ├── dev/
│       ├── test1/
│       └── train/
├── snli-ve
│   ├── snli_ve_dev.jsonl
│   ├── snli_ve_test.jsonl
│   └── snli_ve_train.jsonl
├── vcr
│   ├── annotation
│   │   ├── test.jsonl
│   │   ├── train.jsonl
│   │   └── val.jsonl
│   ├── drawn_images/
│   ├── vcr1images/
└── vqav2
    ├── ans2label.pkl
    ├── v2_mscoco_train2014_annotations.json
    ├── v2_mscoco_val2014_annotations.json
    ├── v2_OpenEnded_mscoco_train2014_questions.json
    └── v2_OpenEnded_mscoco_val2014_questions.json
```

Items ending with `/` are directories, typically containing a large number of images. The `drawn_images` folder for the VCR task can be generated from the original `vcr1images`, using the scripts available [here](https://github.com/rowanz/merlot/tree/main/downstream/vcr/data).
