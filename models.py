import tensorflow as tf


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


class ReflectionPadding2D(tf.keras.layers.Layer):
    '''
      2D Reflection Padding
      Attributes:
        - padding: (padding_width, padding_height) tuple
    '''

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor, [[0, 0], [padding_height, padding_height], [padding_width, padding_width], [0, 0]],
                      'REFLECT')


# replace Conv2DTranspose with upsampling and conv2d, add bias
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()

    result.add(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'))
    result.add(ReflectionPadding2D(padding=(1, 1)))
    result.add(tf.keras.layers.Conv2D(filters, 3, strides=1, padding='valid',
                                      use_bias=True, kernel_initializer=initializer))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def Generator(args):
    inputs = tf.keras.layers.Input(shape=[args.img_h, args.img_w, args.img_c])

    input_hist_1 = tf.keras.layers.Input(shape=[args.bin_num])
    input_hist_2 = tf.keras.layers.Input(shape=[args.bin_num])
    input_hist_3 = tf.keras.layers.Input(shape=[args.bin_num])

    hist_1_em = tf.keras.layers.Embedding(args.img_h * args.img_w + 1, 1, input_length=args.bin_num)(input_hist_1)
    hist_2_em = tf.keras.layers.Embedding(args.img_h * args.img_w + 1, 1, input_length=args.bin_num)(input_hist_2)
    hist_3_em = tf.keras.layers.Embedding(args.img_h * args.img_w + 1, 1, input_length=args.bin_num)(input_hist_3)

    hist_em = tf.keras.layers.Concatenate()([hist_1_em, hist_2_em, hist_3_em])
    hist_em_flat = tf.keras.layers.Flatten()(hist_em)
    hist = tf.keras.layers.Dense(args.bin_num * 2, input_shape=(args.bin_num * 3,), activation='relu')(
        hist_em_flat)

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(args.img_out_c, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # concat input histogram
    hist = tf.expand_dims(hist, 1)  # shape=(None,1, 512)
    hist = tf.expand_dims(hist, 1)  # shape=(None,1,1 512)
    x = tf.keras.layers.Concatenate()([x, hist])  # shape=(None, 1, 1, 1024)

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=[inputs, input_hist_1, input_hist_2, input_hist_3], outputs=x)


def Discriminator(args, conditional):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[args.img_h, args.img_w, args.img_c], name='input_image')
    if not conditional:
        x = inp
        inputs = inp
    else:
        tar = tf.keras.layers.Input(shape=[args.img_h, args.img_w, args.img_c], name='target_image')
        inputs = [inp, tar]
        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inputs, outputs=last)
