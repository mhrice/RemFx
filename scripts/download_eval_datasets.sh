#! /bin/bash

mkdir -p RemFX_eval_datasets
cd RemFX_eval_datasets
mkdir -p processed
cd processed
wget https://zenodo.org/record/8187288/files/0-0.zip?download=1 -O 0-0.zip
wget https://zenodo.org/record/8187288/files/1-1.zip?download=1 -O 1-1.zip
wget https://zenodo.org/record/8187288/files/2-2.zip?download=1 -O 2-2.zip
wget https://zenodo.org/record/8187288/files/3-3.zip?download=1 -O 3-3.zip
wget https://zenodo.org/record/8187288/files/4-4.zip?download=1 -O 4-4.zip
wget https://zenodo.org/record/8187288/files/5-5.zip?download=1 -O 5-5.zip
unzip 0-0.zip
unzip 1-1.zip
unzip 2-2.zip
unzip 3-3.zip
unzip 4-4.zip
unzip 5-5.zip
rm 0-0.zip
rm 1-1.zip
rm 2-2.zip
rm 3-3.zip
rm 4-4.zip
rm 5-5.zip

