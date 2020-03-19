#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

if [ ! -d "pretrained/coco_2014_train+coco_2014_valminusminival" ]; then
  echo "Cannot find COCO models. Please follow instructions in the readme to unpack the pretrained data before running this script."
  exit
fi

echo "Building libraries"
cd lib
make
cd ../data/referit
make
echo "Build complete, unpacking Flickr30K Entities"

cd ../flickr
unzip annotations.zip

wget http://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k-images.tar
tar xvf flickr30k-images.tar
mv flickr30k-images images
rm flickr30k-images.tar

echo "Flickr30K Entities data unpacked, starting ReferIt Game"
cd ..
wget http://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip
unzip refclef.zip
rm refclef.zip

cd referit
wget http://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip
unzip saiapr_tc-12.zip
rm saiapr_tc-12.zip
cd ..

echo "ReferIt Game data unpacked, building vocabularies"

python ./tools/build_vocab.py --remove_stopwords

echo "Flickr30K Entities vocab done, starting ReferIt Game"

python ./tools/build_vocab.py --remove_stopwords --dataset referit

echo "ReferIt Game vocab processing complete"






