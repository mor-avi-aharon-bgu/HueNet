import tensorflow as tf
import imlib as im
import numpy as np


def load_hist(real_image, args):
    hist1 = tf.histogram_fixed_width(real_image[..., 0], [args.min_val, args.max_val], nbins=args.bin_num)
    hist2 = tf.histogram_fixed_width(real_image[..., 1], [args.min_val, args.max_val], nbins=args.bin_num)
    hist3 = tf.histogram_fixed_width(real_image[..., 2], [args.min_val, args.max_val], nbins=args.bin_num)

    return hist1, hist2, hist3


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


def load_colorization(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    image = tf.cast(image, tf.float32)
    gray_image = tf.image.rgb_to_grayscale(image)

    gray_image_3c = tf.keras.layers.concatenate([gray_image, gray_image, gray_image])

    return gray_image_3c, image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


# from rgb {0,..,255} to yuv [-1,1]
def rgb2yuv(img):
    img = tf.image.rgb_to_yuv(img / 255)
    img = tf.stack([img[..., 0] * 2 - 1, img[..., 1] * 2, img[..., 2] * 2], -1)
    return img


# normalizing the images to [-1, 1] (and convert to yuv if needed)
def normalize(input_image, real_image, yuv):
    if yuv:
        input_image = rgb2yuv(input_image)
        real_image = rgb2yuv(real_image)
    else:
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1

    return input_image, real_image


def load_image(image_file, train, args):
    def random_jitter(input_image, real_image):
        def random_crop(input_image, real_image):
            stacked_image = tf.stack([input_image, real_image], axis=0)
            cropped_image = tf.image.random_crop(
                stacked_image, size=[2, args.img_h, args.img_w, args.img_c])

            return cropped_image[0], cropped_image[1]

        # resizing to 286 x 286 x 3
        input_image, real_image = resize(input_image, real_image, 286, 286)

        # randomly cropping to 256 x 256 x 3
        input_image, real_image = random_crop(input_image, real_image)

        if tf.random.uniform(()) > 0.5:
            # random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image

    if args.task == 'colorization':
        input_image, real_image = load_colorization(image_file)
    else:
        input_image, real_image = load(image_file)
    if train and args.jitter:
        input_image, real_image = random_jitter(input_image, real_image)
    else:
        input_image, real_image = resize(input_image, real_image,
                                         args.img_h, args.img_w)
    input_image, real_image = normalize(input_image, real_image, args.yuv)

    if args.a2b:
        return input_image, (real_image, load_hist(real_image, args))
    else:
        return real_image, (input_image, load_hist(input_image, args))


def load_single_image(image_file, hist, train, args):
    def random_jitter_single(image):
        # resizing to 286 x 286 x 3
        image = tf.image.resize(image, [286, 286],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # randomly cropping to 256 x 256 x 3
        image = tf.image.random_crop(image, size=[args.img_h, args.img_w, args.img_c])
        # random mirroring
        image = tf.image.random_flip_left_right(image)
        return image

    # load image
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    # resize or jitter
    if train and args.jitter:
        image = random_jitter_single(image)
    else:
        image = tf.image.resize(image, [args.img_h, args.img_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # normalize
    image = rgb2yuv(image)
    if hist:
        return image, load_hist(image, args)
    else:
        return image


# from yuv [-1,1] to rgb [-1,1]
def yuv2rgb(img):
    img = tf.stack([(img[..., 0] + 1) / 2, img[..., 1] / 2, img[..., 2] / 2], -1)
    img = tf.image.yuv_to_rgb(img)
    img = img * 2 - 1
    return img


def save_images(model, test_input, tar, tar_hist, filename, args):
    prediction = model([test_input, tar_hist[0], tar_hist[1], tar_hist[2]], training=True)
    if args.yuv:
        test_input = yuv2rgb(test_input)
        tar = yuv2rgb(tar)
        prediction = yuv2rgb(prediction)
    img = np.concatenate([test_input[0], tar[0], prediction[0]], axis=1)
    img = tf.clip_by_value(img, -1.0, 1.0).numpy()
    im.imwrite(img, filename)


def shuffle_zip(inp_ds, tar_ds, args, seed_inp=None, seed_tar=None):
    inp_ds = inp_ds.shuffle(buffer_size=args.buffer_size, seed=seed_inp).batch(batch_size=args.batch_size)
    tar_ds = tar_ds.shuffle(buffer_size=args.buffer_size, seed=seed_tar).batch(batch_size=args.batch_size)
    return tf.data.Dataset.zip((inp_ds, tar_ds))
