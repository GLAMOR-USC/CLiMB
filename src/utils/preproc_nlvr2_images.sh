#!/bin/bash

IMAGES_DIR='/data/datasets/MCL/nlvr2/tmp/'

unzip -q ${IMAGES_DIR}/dev_img.zip -d ${IMAGES_DIR}/
rm ${IMAGES_DIR}/dev_img.zip 

unzip -q ${IMAGES_DIR}/test1_img.zip -d ${IMAGES_DIR}
rm ${IMAGES_DIR}/test1_img.zip

unzip -q ${IMAGES_DIR}/train_img.zip -d ${IMAGES_DIR}
src_dir="${IMAGES_DIR}/images/train"
trg_dir="${IMAGES_DIR}/train"
mkdir $trg_dir

for i in {0..99}
do
   mv ${src_dir}/$i/*.png ${trg_dir} 
done

rm -rf $src_dir
rm -rf ${IMAGES_DIR}/images
rm ${IMAGES_DIR}/train_img.zip
