import tensorflow as tf

class BlazeBlock(tf.keras.Model):
    def __init__(self, block_num = 3, channel = 48):
        super(BlazeBlock, self).__init__()
        self.downsample_a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=(2, 2), padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=channel, kernel_size=1, activation=None)
        ])
        self.downsample_b = tf.keras.models.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            # 因为我实在是不会写channel padding的实现，所以这里用了个1x1的卷积来凑个数，嘤~
            tf.keras.layers.Conv2D(filters=channel, kernel_size=1, activation=None)
        ])
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
        self.conv3a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=(2, 2), padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=24, kernel_size=1, activation=None)
        ])
        self.conv3b = tf.keras.models.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            # 因为我实在是不会写channel padding的实现，所以这里用了个1x1的卷积来凑个数
            tf.keras.layers.Conv2D(filters=48, kernel_size=1, activation=None)
        ])

        self.conv3_1 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=48, kernel_size=1, activation=None)
        ])
        self.conv3_2 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=48, kernel_size=1, activation=None)
        ])
        self.conv3_3 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=48, kernel_size=1, activation=None)
        ])

        self.conv4a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=(2, 2), padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=96, kernel_size=1, activation=None)
        ])
        self.conv4b = tf.keras.models.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=96, kernel_size=1, activation=None)
        ])

        self.conv4_1 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=96, kernel_size=1, activation=None)
        ])
        self.conv4_2 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=96, kernel_size=1, activation=None)
        ])
        self.conv4_3 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=96, kernel_size=1, activation=None)
        ])
        self.conv4_4 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=96, kernel_size=1, activation=None)
        ])

    def call(self, x):
        # shape = (1, 256, 256, 3)
        x = self.conv1(x)
        # shape = (1, 128, 128, 24)
        x = x + self.conv2_1(x)   # <-- skip connection
        x = tf.keras.activations.relu(x)
        #   --> I don't know why the relu layer is put after skip connection?
        x = x + self.conv2_2(x)
        x = tf.keras.activations.relu(x)
        y0 = x
        # <----- downsample ----->
        x = tf.keras.activations.relu(self.conv3a(x) + self.conv3b(x))
        # shape = (1, 64, 64, 48)
        x = tf.keras.activations.relu(x + self.conv3_1(x))
        x = tf.keras.activations.relu(x + self.conv3_2(x))
        x = tf.keras.activations.relu(x + self.conv3_3(x))
        y1 = x
        # <----- downsample ----->
        x = tf.keras.activations.relu(self.conv4a(x) + self.conv4b(x))
        # shape = (1, 32, 32, 96)
        x = tf.keras.activations.relu(x + self.conv4_1(x))
        x = tf.keras.activations.relu(x + self.conv4_2(x))
        x = tf.keras.activations.relu(x + self.conv4_3(x))
        x = tf.keras.activations.relu(x + self.conv4_4(x))
        y2 = x
        # <----- downsample ----->
        x = tf.keras.activations.relu(self.conv5a(x) + self.conv5b(x))
        5 res
        y3 = x
        x = tf.keras.activations.relu(self.conv6a(x) + self.conv6b(x))
        6 res
        y4 = x
        return y
