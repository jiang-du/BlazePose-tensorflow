#!~/miniconda3/envs/tf2/bin/python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from model import BlazePose
from config import total_epoch
from data import train_dataset, test_dataset

model = BlazePose()
# optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.MeanSquaredError()])

checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5)

# model.load_weights(checkpoint_path)
model.fit(train_dataset, epochs=total_epoch, verbose=1, callbacks=[cp_callback])
model.save_weights(checkpoint_path.format(epoch=total_epoch))
print("testing on validation set.")
model.evaluate(test_dataset)

model.summary()
