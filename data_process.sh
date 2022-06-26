
!/usr/bin/env bash

cd data

mkdir images
mkdir labels
mkdir txts
mkdir test_images
mkdir xmls

cd ..

python 1_data_process.py
python 2_images2coco.py

