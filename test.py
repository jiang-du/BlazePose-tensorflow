#!~/miniconda3/envs/tf2/bin/python
import os
import tensorflow as tf
from model import BlazePose
from config import total_epoch, train_mode
from data import train_dataset, test_dataset, data

model = BlazePose()
# optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanSquaredError()])

checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# model.evaluate(test_dataset)
model.load_weights(checkpoint_path.format(epoch=199))

"""
0-Right ankle
1-Right knee
2-Right hip
3-Left hip
4-Left knee
5-Left ankle
6-Right wrist
7-Right elbow
8-Right shoulder
9-Left shoulder
10-Left elbow
11-Left wrist
12-Neck
13-Head top
"""

if train_mode:
    import cv2
    import numpy as np
    y = np.zeros((2000, 14, 3)).astype(np.uint8)
    y[0:1000] = model(data[0:1000]).numpy().astype(np.uint8)
    y[1000:2000] = model(data[1000:2000]).numpy().astype(np.uint8)
    for t in range(2000):
        skeleton = y[t]
        print(skeleton)
        img = data[t].astype(np.uint8)
        for i in range(14):
            cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=2, color=(0, 255, 0), thickness=2)
        cv2.imwrite("./result/lsp_%d.jpg"%t, img)
        cv2.imshow("test", img)
        cv2.waitKey(1)
        pass
else:
    model.evaluate(test_dataset)

    y = model.predict(data[1000:1030])

    import matplotlib.pyplot as plt

    for t in range(30):
        plt.figure(figsize=(8,8), dpi=150)
        for i in range(14):
            plt.subplot(4, 4, i+1)
            plt.imshow(y[t, :, :, i])
        plt.savefig("demo.png")
        plt.show()
pass
