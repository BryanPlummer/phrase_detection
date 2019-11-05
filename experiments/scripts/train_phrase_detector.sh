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
    ITERS=380000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  referit)
    TRAIN_IMDB="referit_train"
    TEST_IMDB="referit_val"
    STEPSIZE="[240000]"
    ITERS=380000
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

NET_FINAL=output/${NET}/${TRAIN_IMDB}/${TAG}/${NET}_faster_rcnn_iter_${ITERS}.ckpt
CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
    --weight pretrained/coco_2014_train+coco_2014_valminusminival/${NET}_faster_rcnn_iter_1190000.ckpt \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${TEST_IMDB} \
    --iters ${ITERS} \
    --cfg experiments/cfgs/${NET}.yml \
    --tag ${TAG} \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
    TRAIN.STEPSIZE ${STEPSIZE}

sh test_phrase_detector.sh ${GPU_ID} ${DATASET} ${NET} ${TAG}
