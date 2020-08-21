#!~/miniconda3/envs/tf2/bin/python
import os
import tensorflow as tf
from model import BlazePose
from config import total_epoch
from data import train_dataset, test_dataset

model = BlazePose()
# optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanSquaredError()])

checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=5)

model.fit(train_dataset, epochs=total_epoch, callbacks=[cp_callback])
print("testing on validation set.")
model.evaluate(test_dataset)

model.summary()
