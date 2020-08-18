import tensorflow as tf

class BlazeBlock(tf.keras.Model):
    def __init__(self, block_num = 3, channel = 48):
        super(BlazeBlock, self).__init__()
        # <----- downsample ----->
        self.downsample_a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=(2, 2), padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=channel, kernel_size=1, activation=None)
        ])
        self.downsample_b = tf.keras.models.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            # 因为我实在是不会写channel padding的实现，所以这里用了个1x1的卷积来凑个数，嘤~
            tf.keras.layers.Conv2D(filters=channel, kernel_size=1, activation=None)
        ])
        # <----- separable convolution ----->
        self.conv = list()
        for i in range(block_num):
            self.conv.append(tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=channel, kernel_size=1, activation=None)
        ]))

    def call(self, x):
        x = tf.keras.activations.relu(self.downsample_a(x) + self.downsample_b(x))
        for conv_layer in range(self.conv):
            x = tf.keras.activations.relu(x + conv_layer(x))
        return x

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

        # heatmap encoder
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
            tf.keras.layers.Conv2D(filters=1, kernel_size=3, activation=None)
        ])

        # regression branch
        self.conv12a = BlazeBlock(block_num = 4, channel = 96)    # input res: 64
        self.conv12b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=96, kernel_size=1, activation="relu")
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
        heatmap = self.conv11(y)

        # regression branch
        x = self.conv12(x) + self.conv12b
        # shape = (1, 32, 32, 96)
        return heatmap
