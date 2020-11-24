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

## Train (from random initialization)

1. Download LSP dataset. (If you already have, just skip this step)

    If you are the first time to run this code on Linux platform, LSP dataset will be downloaded.
    
    However, if you are using Microsoft Windows 10, please download and unzip the dataset manually.

2. Pre-train the heatmap branch.

    Edit training settings in `config.py`. Set `train_mode = 0` and `continue_train = 0`.

    Then, run `python3 train.py`.

3. Fine-tune for the joint regression branch.

    Set `train_mode = 1`, `continue_train = 0` and `best_pre_train` with the num of epoch where the training loss drops but testing accuracy achieve the optimal.

    Then, run `python3 train.py`.

## Continue training

If you have just suffered from an unexpectedly power off or press `Ctrl + C` to cut up training, and then you want to continue your work, follow the following steps:

1. Edit `config.py`, modify `continue_train` to the epoch where you want to start with.

    For continue pre-train, simply set the value of `continue_train`.
    
    For fine-tune, set `train_mode = 1`, and `continue_train` to num of epoch where the training loss drops but testing accuracy achieve the optimal.

2. Run `python3 train.py`.

3. If you are running pre-train just now, after that, just set `train_mode = 1`, `continue_train = 0` and `best_pre_train` with the num of epoch where the training loss drops but testing accuracy achieve the optimal, and run `python3 train.py`.

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

    - [x] Two stage training (pre-train and fine-tune).

    - [x] Continue training from a custom epoch of checkpoint.

    - [x] Save the training record (loss and accuracy for training and validation set) to json file.

    - [x] More explicit training settings (for fine-tune and continue training)

- [ ] Dataset and preprocess.

    - [x] LSP dataset train and validation.

    - [ ] LSPET dataset.

    - [ ] Custom dataset.

- [ ] Implementation of pose tracking on video.

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
