# A Tensorflow Implementation for BlazePose

This is a third-party tensorflow implementation for BlazePose.

The original paper is "BlazePose: On-device Real-time Body Pose tracking" by Valentin Bazarevsky, Ivan Grishchenko, Karthik Raveendran, Tyler Zhu, Fan Zhang, and Matthias Grundmann. Available on [arXiv](https://arxiv.org/abs/2006.10204).

Since I do not have the full settings provided by the original author. There might be something different from the original paper. Please forgive me if I write something wrong.

Works are in process. The current version does not stand for the full functions.

## Requirements

It is highly recommented to run this code on Ubuntu 20.04 with Anaconda environment.

```
python >= 3.8.5
tensorflow >= 2.3
numpy
matplotlib
```

## Train

1. Modify training settings in `config.py`.

2. Run `python3 train.py`.

3. If you are the first time to run this code, LSP dataset will be downloaded. Especially, if you are using Microsoft Windows 10, please download and unzip the dataset manually.

## Test

1. Run `python3 test.py`.

## TODOs

- [x] Basic code for network model BlazePose.

- [ ] Dataset and preprocess.

    - [x] LSP dataset train and validation.

    - [ ] LSPET dataset.

    - [ ] Custom dataset.

- [ ] Two stage training.

- [ ] Online camera demo.

## Comments

Please feel free to submit an issue or pull a request.
