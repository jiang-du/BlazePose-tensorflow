# A Tensorflow Implementation for BlazePose

This is a third-party TensorFlow implementation for BlazePose.

The original paper is "BlazePose: On-device Real-time Body Pose tracking" by Valentin Bazarevsky, Ivan Grishchenko, Karthik Raveendran, Tyler Zhu, Fan Zhang, and Matthias Grundmann. Available on [arXiv](https://arxiv.org/abs/2006.10204).

Since I do not have the full settings provided by the original author. There might be something different from the original paper. Please forgive me if I write something wrong.

Works are in process. The current version does not stand for the full functions.

## Requirements

It is highly recommended to run this code on Ubuntu 20.04 with an Anaconda environment. Python 3.7.9 and 3.8.5 has been tested. CUDA version has been tested with `10.1` and `11.1`.

```
tensorflow >= 2.3
numpy
matplotlib
scipy
```

## Train (from random initialization)

1. Download LSP dataset. (If you already have, just skip this step)

    If you are the first time to run this code on Linux platform, the LSP dataset will be downloaded.
    
    However, if you are using Microsoft Windows 10, please download and unzip the dataset manually.

2. Pre-train the heatmap branch.

    Edit training settings in `config.py`. Set `train_mode = 0` and `continue_train = 0`.

    Then, run `python3 train.py`.

3. Fine-tune for the joint regression branch.

    Set `train_mode = 1`, `continue_train = 0` and `best_pre_train` with the num of epoch where the training loss drops but testing accuracy achieve the optimal.

    Then, run `python3 train.py`.

## Continue training

If you have just suffered from an unexpected power off or press `Ctrl + C` to cut up training, and then you want to continue your work, follow the following steps:

1. Edit `config.py`, modify `continue_train` to the epoch where you want to start with.

    For continue pre-train, simply set the value of `continue_train`.
    
    For fine-tuning, set `train_mode = 1`, and `continue_train` to num of epoches where the training loss drops but testing accuracy achieve the optimal.

2. Run `python3 train.py`.

3. If you are running pre-train just now, after that, just set `train_mode = 1`, `continue_train = 0`, and `best_pre_train` with the num of epoches where the training loss drops but testing accuracy achieve the optimal, and run `python3 train.py`.

## Test

1. Edit `config.py`.

    If you want to see the visualized heatmaps, set `train_mode = 0`.
  
    For skeleton joint results, set `train_mode = 1`.

2. Set `epoch_to_test` to the epoch you would like to test.

3. If you set `train_mode = 0`, you should also set `vis_img_id` to select an image.

4. For `train_mode = 1`, evaluation mode should be set.

    Set `eval_mode = 1` if you want to get PCKh@0.5 score, or `eval_mode = 0` if you want to get the result images.

5. If you are the first time to set `train_mode = 1` and `eval_mode = 0`, open terminal:

    ```bash
    mkdir result
    ```

6. Run `python3 test.py`.

    For `train_mode = 0`, you will see the heatmap.

    For `train_mode = 1` and `eval_mode = 0`, the tested images will be written in `result` dictionary.

    For `train_mode = 1` and `eval_mode = 1`, PCKh@0.5 scores of each joint and the average score will be shown.

## Online camera demo

1. Install YOLOv4 package using `pip3 install yolov4`. The trained model of YOLOv4 should also be downloaded.

2. Finish training on your dataset.

3. Set `train_mode = 1` and connect to a USB camera.

4. Run `python3 demo.py`. You should allow one or a few person(s) standing in front of the camera.

## TODOs

- [x] Basic code for network model BlazePose.

    - [x] Implementation of Channel Attention layer.

- [x] Functions

    - [x] Two-stage training (pre-train and fine-tune).

    - [x] Continue training from a custom epoch of checkpoint.

    - [x] Save the training record (loss and accuracy for training and validation set) to json file.

    - [x] More explicit training settings (for fine-tune and continue training).
    
    - [x] Calculate PCKh@0.5 scores.

- [ ] Dataset and preprocess.

    - [x] LSP dataset train and validation.

    - [ ] LSPET dataset.

    - [ ] Custom dataset.

- [ ] Implementation of pose tracking on video.

- [x] Online camera demo.

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
