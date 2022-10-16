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

Items ending with `/` are directories, typically containing a large number of images. 

For NLVR2:
- the link for downloading images can be requested using [this form](https://docs.google.com/forms/d/e/1FAIpQLSdB_OhgmpQULV17kjQ4iitftILbOJjuGgJ2ECmg-HdmkjUSAg/viewform). 
- Download the three zip files (`train_img.zip`, `dev_img.zip`, `test_img.zip`) using `wget` into `nlvr2/images/`.
- Run `bash src/utils/preproc_nlvr2_images.sh`, with the `IMAGES_DIR` variable set to the full path for `nlvr2/images/`.
- The files in `nlvr2/data/` can be downloaded from the [NLVR2 GitHub repo](https://github.com/lil-lab/nlvr/tree/master/nlvr2/data).

The `drawn_images` folder for the VCR task can be generated from the original `vcr1images`, using the scripts available [here](https://github.com/rowanz/merlot/tree/main/downstream/vcr/data).

## Language-Only Tasks

CLiMB initially includes five language-only tasks: 
- [IMDb](https://huggingface.co/datasets/imdb)
- [SST-2](https://huggingface.co/datasets/glue/viewer/sst2)
- [PIQA](https://yonatanbisk.com/piqa/data/)
- [HellaSwag](https://github.com/rowanz/hellaswag/tree/master/data)
- [CommonsenseQA](https://www.tau-nlp.sites.tau.ac.il/commonsenseqa)

We provide the script ```utils/download_lang_mc.sh``` for downloading multiple-choice tasks from the official websites linked above.
*Note:* we split our dev set (held-out) from the training set for hyper-parameter tuning and use the original dev set as the test set, as we do not have the labels of the original test set.

## Vision-Only Tasks

CLiMB initially includes four vision-only tasks: 
- [ImageNet-1000](https://image-net.org/download.php)
- [iNaturalist 2019](https://github.com/visipedia/inat_comp/tree/master/2019)
- [Places365](http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar)
- [COCO multi-label object classification](https://cocodataset.org/#download)


Data files for these four tasks can be downloaded from their respective websites. The data files are organized as follow:

```
YOUR_DATA_DIR
├── ILSVRC2012/
|   ├── train/
|   |   ├── n01440764/
|   |   ├── n01443537/
|   |   └── ...
|   ├── val/
|   |   └── ILSVRC2012_val_*.JPEG
|   └── LOC_val_solution.csv
|── iNat2019/
|   ├── train_val2019/
|   |   ├── Amphibians/
|   |   ├── Birds/
|   |   └── ...
|   ├── train2019.json
|   └── val2019.json
|── Places365/
|   ├── train/
|   |   ├── airfield/
|   |   ├── airplane_cabin/
|   |   └── ...
|   └── val/
|       ├── airfield/
|       ├── airplane_cabin/
|       └── ...
└── ms-coco/
    ├── images/
    └── detections/
        └── annotations/
            ├── instances_train2017.json
            └── instances_val2017.json
```
*Note:* we split our dev set (held-out) from the training set for hyper-parameter tuning and use the original dev set as the test set, as we do not have the labels of the original test set.
