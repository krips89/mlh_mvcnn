# MVCNN for Multi-Layered Height-maps

This repo contains the training and testing code for classification using a variation of Multi-view CNN (that uses non comutative merge operation) and *Multi-Layered Height-map* features of 3D shapes. The details are available in the following paper which is to be presented at ECCV 2018:

```
Sarkar, Kripasindhu, Basavaraj Hampiholi, Kiran Varanasi, and Didier Stricker. 
"Learning 3D Shapes as Multi-Layered Height-maps using 2D Convolutional Networks." 
In Proceedings of the European Conference on Computer Vision (ECCV), pp. 71-86. 2018.
```
Bibtex - 
 
 ```
@InProceedings{Sarkar_2018_ECCV,
author = {Sarkar, Kripasindhu and Hampiholi, Basavaraj and Varanasi, Kiran and Stricker, Didier},
title = {Learning 3D Shapes as Multi-Layered Height-maps using 2D Convolutional Networks},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}
```

Please find the Arxiv preprint of the paper [here](https://arxiv.org/abs/1807.08485).

# Getting started

## Dependencies
This code is based on **PyTorch**. Just get the latest version of pytorch as per the official website (https://pytorch.org/get-started/locally/). Or run the following command (requires pip) in the shell:

```
pip install torch torchvision
```
## Steps for getting started
1. Clone this repository (lets say to MLH_MVCNN_ROOT).
2. Download the MLH features of **ModelNet40** [here](http://www.dfki.uni-kl.de/~sarkar/ML_MN_int_256_5l_3v.zip) and extract it to <mlh_root_path>.
3. Edit the `train_data_root` variable in config.py to point to <mlh_root_path>.

# Training
Simply run `python train.py` to train Multi-View CNN with non-commutative merge operation (for details see the paper) with MLH descriptors. Edit the training parameters in `config.py` to further control the training. Training for 20 epoches should give a validation accuracy of around 93.1.

# Testing 
* Run `python test.py <path_to_saved_model>` to test your trained model. By default, the best model gets saved to `MLH_MVCNN_ROOT/data/vgg16_bn_best_model.pth.tar` (configurable through config.py) while training.
* We are also providing the trained model used in our paper [here](http://www.dfki.uni-kl.de/~sarkar/vgg16_bn_paper_model.pth.tar). Just download it and use it as <path_to_saved_model> to get the testing results on ModelNet40.

