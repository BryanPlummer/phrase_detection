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
    MAX_PHRASES=5
    ;;
  referit)
    TRAIN_IMDB="referit_train"
    TEST_IMDB="referit_val"
    STEPSIZE="[160000]"
    ITERS=200000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    MAX_PHRASES=2
    ;;
  vg)
    TRAIN_IMDB="vg_train"
    TEST_IMDB="vg_val"
    STEPSIZE="[350000]"
    ITERS=490000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    MAX_PHRASES=5
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${TAG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"



NET_FINAL=output/${NET}/${TRAIN_IMDB}/${TAG}/${NET}_iter_${ITERS}.ckpt


cd external/cite
CCA_PARAMS=../../output/${NET}/${TRAIN_IMDB}/${TAG}/${NET}_iter_${ITERS}/cca_parameters.pkl
CUDA_VISIBLE_DEVICES=${GPU_ID} time python main.py \
    --spatial --use_augmented --dataset ${DATASET} --name ${TAG} \
    --datadir data/${DATASET}/${TAG} --cca_parameters ${CCA_PARAMS} \
    --word_embedding ../../data/hglmm_6kd.txt --region_norm_axis 2 \
    --max_phrases 40 --ifs
cd ../..

./experiments/scripts/test_augmented_phrase_detector.sh ${GPU_ID} ${DATASET} ${NET} ${TAG}
