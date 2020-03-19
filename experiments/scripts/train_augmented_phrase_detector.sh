#!/bin/bash

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3
TAG=$4

case ${DATASET} in
  flickr)
    TRAIN_IMDB="flickr_train"
    TEST_IMDB="flickr_val"
    STEPSIZE="[240000]"
    ITERS=360000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  referit)
    TRAIN_IMDB="referit_train"
    TEST_IMDB="referit_val"
    STEPSIZE="[80000]"
    ITERS=100000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${TAG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
    --weight pretrained/coco_2014_train+coco_2014_valminusminival/${NET}_faster_rcnn_iter_1190000.ckpt \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${TEST_IMDB} \
    --iters ${ITERS} \
    --cfg experiments/cfgs/${NET}.yml \
    --tag ${TAG} \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
    TRAIN.STEPSIZE ${STEPSIZE} AUGMENTED_POSITIVE_PHRASES True

NET_FINAL=output/${NET}/${TRAIN_IMDB}/${TAG}/${NET}_iter_${ITERS}.ckpt
CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/compute_cca.py \
    --imdb ${TRAIN_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --tag ${TAG} \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
    AUGMENTED_POSITIVE_PHRASES True

./data/scripts/cache_augmented_cite_features.sh ${GPU_ID} ${DATASET} ${NET} ${TAG}

cd external/cite
CCA_PARAMS=../../output/${NET}/${TRAIN_IMDB}/${TAG}/${NET}_iter_${ITERS}/cca_parameters.pkl
CUDA_VISIBLE_DEVICES=${GPU_ID} time python main.py \
    --spatial --use_augmented --dataset ${DATASET} --name ${TAG} \
    --datadir data/${DATASET}/${TAG} --cca_parameters ${CCA_PARAMS} \
    --word_embedding ../../data/hglmm_6kd.txt --region_norm_axis 2 \
    --max_phrases 40 --ifs
cd ../..

./experiments/scripts/test_augmented_phrase_detector.sh ${GPU_ID} ${DATASET} ${NET} ${TAG}
