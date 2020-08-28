import tensorflow as tf
from layers import BlazeBlock
from config import num_joints, train_mode

class BlazePose(tf.keras.Model):
    def __init__(self):
        super(BlazePose, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=24, kernel_size=3, strides=(2, 2), padding='same', activation='relu'
        )
         
        # separable convolution (MobileNet)
        self.conv2_1 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=24, kernel_size=1, activation=None)
        ])
        self.conv2_2 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=24, kernel_size=1, activation=None)
        ])

        #  ---------- Heatmap branch ----------
        self.conv3 = BlazeBlock(block_num = 3, channel = 48)    # input res: 128
        self.conv4 = BlazeBlock(block_num = 4, channel = 96)    # input res: 64
        self.conv5 = BlazeBlock(block_num = 5, channel = 192)   # input res: 32
        self.conv6 = BlazeBlock(block_num = 6, channel = 288)   # input res: 16

        self.conv7a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=48, kernel_size=1, activation="relu"),
            tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        ])
        self.conv7b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=48, kernel_size=1, activation="relu")
        ])

        self.conv8a = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        self.conv8b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=48, kernel_size=1, activation="relu")
        ])

        self.conv9a = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        self.conv9b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=48, kernel_size=1, activation="relu")
        ])

        self.conv10a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=8, kernel_size=1, activation="relu"),
            tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        ])
        self.conv10b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=8, kernel_size=1, activation="relu")
        ])

        # the output layer for heatmap and offset
        self.conv11 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=8, kernel_size=1, activation="relu"),
            # heatmap
            tf.keras.layers.Conv2D(filters=num_joints, kernel_size=3, padding="same", activation=None)
        ])

        # ---------- Regression branch ----------
        #  shape = (1, 64, 64, 48)
        self.conv12a = BlazeBlock(block_num = 4, channel = 96)    # input res: 64
        self.conv12b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=96, kernel_size=1, activation="relu")
        ])

        self.conv13a = BlazeBlock(block_num = 5, channel = 192)   # input res: 32
        self.conv13b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=192, kernel_size=1, activation="relu")
        ])

        self.conv14a = BlazeBlock(block_num = 6, channel = 288)   # input res: 16
        self.conv14b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=288, kernel_size=1, activation="relu")
        ])

        self.conv15 = tf.keras.models.Sequential([
            BlazeBlock(block_num = 7, channel = 288, channel_padding = 0),
            BlazeBlock(block_num = 7, channel = 288, channel_padding = 0)
        ])

        self.conv16 = tf.keras.models.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            # shape = (1, 1, 1, 288)
            tf.keras.layers.Dense(units=3*num_joints, activation=None),
            tf.keras.layers.Reshape((num_joints, 3))
        ])

    def call(self, x):
        # shape = (1, 256, 256, 3)
        x = self.conv1(x)
        # shape = (1, 128, 128, 24)
        x = x + self.conv2_1(x)   # <-- skip connection
        x = tf.keras.activations.relu(x)
        #   --> I don't know why the relu layer is put after skip connection?
        x = x + self.conv2_2(x)
        y0 = tf.keras.activations.relu(x)

        # shape = (1, 128, 128, 24)
        y1 = self.conv3(y0)
        y2 = self.conv4(y1)
        y3 = self.conv5(y2)
        y4 = self.conv6(y3)
        # shape = (1, 8, 8, 288)

        x = self.conv7a(y4) + self.conv7b(y3)
        x = self.conv8a(x) + self.conv8b(y2)
        # shape = (1, 32, 32, 96)
        x = self.conv9a(x) + self.conv9b(y1)
        # shape = (1, 64, 64, 48)
        y = self.conv10a(x) + self.conv10b(y0)
        # shape = (1, 128, 128, 8)
        heatmap = tf.keras.activations.sigmoid(self.conv11(y))

        # ---------- regression branch ----------
        x = self.conv12a(x) + self.conv12b(y2)
        # shape = (1, 32, 32, 96)
        x = self.conv13a(x) + self.conv13b(y3)
        # shape = (1, 16, 16, 192)
        x = self.conv14a(x) + self.conv14b(y4)
        # shape = (1, 8, 8, 288)
        x = self.conv15(x)
        # shape = (1, 2, 2, 288)
        joints = self.conv16(x)
        result = [heatmap, joints]
        return result[train_mode] # heatmap, joints
