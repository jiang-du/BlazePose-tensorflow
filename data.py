#!~/miniconda3/envs/tf2/bin/python
import os
import platform
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from config import num_joints, batch_size, gaussian_sigma, gpu_dynamic_memory

DataURL = "https://sam.johnson.io/research/lsp_dataset.zip"

# guassian generation
def getGaussianMap(joint = (16, 16), heat_size = 128, sigma = 2):
    # by default, the function returns a gaussian map with range [0, 1] of typr float32
    heatmap = np.zeros((heat_size, heat_size),dtype=np.float32)
    tmp_size = sigma * 3
    ul = [int(joint[0] - tmp_size), int(joint[1] - tmp_size)]
    br = [int(joint[0] + tmp_size + 1), int(joint[1] + tmp_size + 1)]
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma ** 2)))
    g.shape
    # usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], heat_size) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], heat_size) - ul[1]
    # image range
    img_x = max(0, ul[0]), min(br[0], heat_size)
    img_y = max(0, ul[1]), min(br[1], heat_size)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    """
    heatmap *= 255
    heatmap = heatmap.astype(np.uint8)
    cv2.imshow("debug", heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    return heatmap

if gpu_dynamic_memory:
    # Limit GPU memory usage if necessary
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

if not os.path.exists("./dataset"):
    os.system("mkdir dataset")

if os.path.exists("./dataset/lsp/joints.mat"):
    print("Found lsp dataset.")
else:
    if not os.path.isfile("./dataset/lsp_dataset.zip"):
        # try to download
        if os.system("wget " + DataURL):
            # abnormal when download with "wget"
            if platform.system() == 'Linux':
                # 下载失败，没有网，训练个锤子啊
                raise Exception("No Internet. Would you like to train with your hammer?")
            elif platform.system() == 'Windows':
                raise Exception('You should firstly install "wget" if you run on Windows.')
            else:
                raise Exception("Unsupported platform. Please run on Ubuntu 20.04.")
        else:
            print("Finish download lsp dataset.")
        
        # try to uncompress
        if platform.system() == 'Linux':
            if os.system("unzip dataset/lsp_dataset.zip -d dataset/lsp/"):
                raise Exception("Unzip Runtime Error. You can try 'rm ./dataset/lsp_dataset.zip' and run again.")
            else:
                print("Finish uncompress lsp dataset.")
        elif platform.system() == 'Windows':
            raise Exception('Please unzip files manually on windows.')
        else:
            raise Exception("Unsupported platform. Please run on Ubuntu 20.04.")
        print("Finish uncompress lsp dataset.")

# read annotations
annotations = loadmat("./dataset/lsp/joints.mat")
label = annotations["joints"].swapaxes(0, 2)    # shape (3, 14, 2000) -> (2000, 14, 3)

# read images
data = np.zeros([2000, 256, 256, 3])
heatmap_set = np.zeros((2000, 128, 128, num_joints), dtype=np.float32)
print("Reading dataset...")
for i in range(2000):
    FileName = "./dataset/lsp/images/im%04d.jpg" % (i + 1)
    img = tf.io.read_file(FileName)
    img = tf.image.decode_image(img)
    img_shape = img.shape
    # Attention here img_shape[0] is height and [1] is width
    label[i, :, 0] *= (256 / img_shape[1])
    label[i, :, 1] *= (256 / img_shape[0])
    data[i] = tf.image.resize(img, [256, 256])
    # generate heatmap set
    for j in range(num_joints):
        _joint = (label[i, j, 0:2] // 2).astype(np.uint16)
        # print(_joint)
        heatmap_set[i, :, :, j] = getGaussianMap(joint = _joint, heat_size = 128, sigma = gaussian_sigma)
    # print status
    if not i%(2000//80):
        print(">", end='')

# dataset
print("\nGenerating training and testing data batches...")
train_dataset = tf.data.Dataset.from_tensor_slices((data[0:1000], heatmap_set[0:1000]))
test_dataset = tf.data.Dataset.from_tensor_slices((data[1000:-1], heatmap_set[1000:-1]))

SHUFFLE_BUFFER_SIZE = 1000
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Finetune
finetune_train = tf.data.Dataset.from_tensor_slices((data[0:1000], label[0:1000]))
finetune_validation = tf.data.Dataset.from_tensor_slices((data[1000:-1], label[1000:-1]))

finetune_train = finetune_train.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
finetune_validation = finetune_validation.batch(batch_size)

print("Done.")
