#!~/miniconda3/envs/tf2/bin/python
import os
import tensorflow as tf
from model import BlazePose
from config import total_epoch
from data import train_dataset, test_dataset, data

model = BlazePose()
# optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanSquaredError()])

checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# model.evaluate(test_dataset)
model.load_weights(checkpoint_path.format(epoch=total_epoch))

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
