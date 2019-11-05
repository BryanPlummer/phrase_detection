# phrase_detection
A Tensorflow implementation of phrase detection framework by Bryan Plummer (bplum@bu.edu). This repository is based on the tensorflow implementation of Faster R-CNN available [here](https://github.com/endernewton/tf-faster-rcnn) which in turn was based on the python Caffe implementation of Faster RCNN available [here](https://github.com/rbgirshick/py-faster-rcnn).

### Prerequisites
  - A basic Tensorflow installation. The code follows **r1.2** format. 
  - Python packages you might not have: `nltk` `cython` `opencv-python` `easydict==1.6` `scikit-image` `pyyaml`

Code was tested using python 2.7

### Installation
1. Clone the repository
  ```Shell
  git clone --recursive https://github.com/BryanPlummer/phrase_detection.git
  ```
  We shall refer to the repo's root directory as `$ROOTDIR`

2. Update your -arch in setup script to match your GPU
  ```Shell
  cd $ROOTDIR/lib
  # Change the GPU architecture (-arch) if necessary
  vim setup.py
  ```

3. Download and unpack the [Flickr30K Entities](http://bryanplummer.com/Flickr30kEntities/) and [ReferIt Game](http://tamaraberg.com/referitgame/) datasets and build the modules and vocabularies from `$ROOTDIR` using,
  ```Shell
  ./data/scripts/fetch_datasets.sh
  ```
  
4. Download pretrained [COCO models](https://drive.google.com/drive/folders/0B1_fAEgxdnvJSmF3YUlZcHFqWTQ) of the desired network which were released in [this repo](https://github.com/endernewton/tf-faster-rcnn).  Dy default, the code assumes they have been unpacked in a directory called `pretrained`.

5. Download a pretrained word embedding.  By default, the code assumes you have downloaded the HGLMM 6K-D vectors from [here](http://ai.bu.edu/grovle/) and placed the unziped file in the `data` directory.  If you want to use a different word embedding, please update the pointer to the embedding file and its dimensions in `lib/model/config.py`.

### Train your own model
Assuming you completed the Installation setup correctly, you should be able to train a model with,
  ```Shell
  ./experiments/scripts/train_phrase_detector.sh [GPU_ID] [DATASET] [NET] [TAG]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {flickr, referit} is defined in train_phrase_detector.sh
  # TAG is an experiment name
  # Examples:
  ./experiments/scripts/train_phrase_detector.sh 0 flickr res101 default
  ./experiments/scripts/train_phrase_detector.sh 1 referit res101 default
  ```
  

### Test and evaluate
You can test your models using,
  ```Shell
  ./experiments/scripts/test_phrase_detector.sh [GPU_ID] [DATASET] [NET] [TAG]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {flickr, referit} is defined in test_phrase_detector.sh
  # TAG is an experiment name
  # Examples:
  ./experiments/scripts/test_phrase_detector.sh 0 flickr res101 default
  ./experiments/scripts/test_phrase_detector.sh 1 referit res101 default
  ```

By default, trained networks are saved under:

```
output/[NET]/[DATASET]/{TAG}/
```

### Citation
If you find our code useful please consider citing:

    @article{plummerPhrasedetection,
      title={Revisiting Image-Language Networks for Open-ended Phrase Detection},
      author={Bryan A. Plummer and Kevin J. Shih and Yichen Li and Ke Xu and Svetlana Lazebnik and Stan Sclaroff and Kate Saenko},
      journal={arXiv:1811.07212},
      year={2018}
    }
    

