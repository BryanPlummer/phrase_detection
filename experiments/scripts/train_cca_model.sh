#!/bin/bash

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3
TAG=$4

case ${DATASET} in
  flickr)
    TRAIN_IMDB="flickr_train"
    TEST_IMDB="flickr_test"
    STEPSIZE="[300000]"
    ITERS=360000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  referit)
    TRAIN_IMDB="referit_train"
    TEST_IMDB="referit_test"
    STEPSIZE="[160000]"
    ITERS=360000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/test_${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

NET_FINAL=output/${NET}/${TRAIN_IMDB}/${TAG}/${NET}_faster_rcnn_iter_${ITERS}.ckpt
CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/compute_deepcca.py \
    --imdb ${TRAIN_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --tag ${TAG} \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS}


