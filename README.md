# A Tensorflow Implementation for BlazePose

This is a third-party tensorflow implementation for BlazePose.

The original paper is "BlazePose: On-device Real-time Body Pose tracking" by Valentin Bazarevsky, Ivan Grishchenko, Karthik Raveendran, Tyler Zhu, Fan Zhang, and Matthias Grundmann. Available on [arXiv](https://arxiv.org/abs/2006.10204).

Since I do not have the full settings provided by the original author. There might be something different from the original paper. Please forgive me if I write something wrong.

Works are in process. The current version does not stand for the full functions.

## Requirements

It is highly recommented to run this code on Ubuntu 20.04 with Anaconda environment. Python 3.7.9 and 3.8.5 has beed tested. CUDA version has been tested with `10.1` and `11.1`.

```
tensorflow >= 2.3
numpy
matplotlib
scipy
```

## Train from random initialization

1. Modify training settings in `config.py`.

    For pre-train of heatmap, we should set `train_mode = 0`.

2. Run `python3 train.py`.

3. If you are the first time to run this code, LSP dataset will be downloaded. Especially, if you are using Microsoft Windows 10, please download and unzip the dataset manually.

## Continue training

1. Edit `config.py`, modify `continue_train` to the epoch where you want to start with.

    For continue pre-train, simply set the value of `continue_train`.
    
    For fine-tune, set `train_mode = 1`, and `continue_train` to num of epoch where the training loss drops but testing accuracy achieve the optimal.

2. Run `python3 train.py`.

## Test

1. Modify training settings in `config.py`.

    If you want to see the visualized heatmaps, set `train_mode = 0`.
  
    For skeleton joint results, set `train_mode = 1`.

2. If you are the first time to test, open terminal:

    ```bash
    mkdir result
    ```

3. Run `python3 test.py`.

    The tested images will be written in `result` dictionary.

## TODOs

- [x] Basic code for network model BlazePose.

    - [x] Implementation of Channel Attention layer.

- [ ] Functions

    - [x] Continue training from a custom epoch of checkpoint.

    - [x] Save the training record (loss and accuracy for training and validation set) to json file.

    - [ ] More explicit training settings (for fine-tune)

- [ ] Dataset and preprocess.

    - [x] LSP dataset train and validation.

    - [ ] LSPET dataset.

    - [ ] Custom dataset.

- [x] Two stage training (pre-train and fine-tune).

- [ ] Online camera demo.

## Reference

If the original paper helps your research, you can cite this paper in the LaTex file with:

```tex
@article{Bazarevsky2020BlazePoseOR,
  title={BlazePose: On-device Real-time Body Pose tracking},
  author={Valentin Bazarevsky and I. Grishchenko and K. Raveendran and Tyler Lixuan Zhu and Fangfang Zhang and M. Grundmann},
  journal={ArXiv},
  year={2020},
  volume={abs/2006.10204}
}
```

## Comments

Please feel free to [submit an issue](https://github.com/jiang-du/BlazePose-tensorflow/issues) or [pull a request](https://github.com/jiang-du/BlazePose-tensorflow/pulls).
