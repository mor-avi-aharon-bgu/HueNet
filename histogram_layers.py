import tensorflow as tf
import numpy as np


class HistogramLayers(object):
    """Network augmentation for 1D and 2D (Joint) histograms construction,
    Calculate Earth Mover's Distance, Mutual Information loss
    between output and target
    """

    def __init__(self, out, tar, args):
        self.bin_num = args.bin_num
        self.min_val = args.min_val
        self.max_val = args.max_val
        self.interval_length = (self.max_val - self.min_val) / self.bin_num
        self.kernel_width = self.interval_length / args.kernel_width_ratio
        self.maps_out = self.calc_activation_maps(out)
        self.maps_tar = self.calc_activation_maps(tar)
        self.n_pixels = self.maps_out.get_shape().as_list()[1]  # number of pixels in image (H*W)
        self.bs = self.maps_out.get_shape().as_list()[0]  # batch size

    def calc_activation_maps(self, img):
        # apply approximated shifted rect (bin_num) functions on img
        bins_min_max = np.linspace(self.min_val, self.max_val, self.bin_num + 1)
        bins_av = (bins_min_max[0:-1] + bins_min_max[1:]) / 2
        bins_av = tf.constant(bins_av, dtype=tf.float32)  # shape = (,bin_num)
        bins_av = tf.expand_dims(bins_av, axis=0)  # shape = (1,bin_num)
        bins_av = tf.expand_dims(bins_av, axis=0)  # shape = (1,1,bin_num)
        img_flat = tf.expand_dims(tf.keras.layers.Flatten()(img), axis=-1)
        maps = self.activation_func(img_flat, bins_av)  # shape = (batch_size,H*W,bin_num)
        return maps

    def activation_func(self, img_flat, bins_av):
        img_minus_bins_av = tf.subtract(img_flat, bins_av)  # shape=  (batch_size,H*W,bin_num)
        img_plus_bins_av = tf.add(img_flat, bins_av)  # shape = (batch_size,H*W,bin_num)
        maps = tf.math.sigmoid((img_minus_bins_av + self.interval_length / 2) / self.kernel_width) \
               - tf.math.sigmoid((img_minus_bins_av - self.interval_length / 2) / self.kernel_width) \
               + tf.math.sigmoid((img_plus_bins_av - 2 * self.min_val + self.interval_length / 2) / self.kernel_width) \
               - tf.math.sigmoid((img_plus_bins_av - 2 * self.min_val - self.interval_length / 2) / self.kernel_width) \
               + tf.math.sigmoid((img_plus_bins_av - 2 * self.max_val + self.interval_length / 2) / self.kernel_width) \
               - tf.math.sigmoid((img_plus_bins_av - 2 * self.max_val - self.interval_length / 2) / self.kernel_width)
        return maps

    def calc_cond_entropy_loss(self, maps_x, maps_y):
        pxy = tf.matmul(maps_x, maps_y, transpose_a=True) / self.n_pixels
        py = tf.reduce_sum(pxy, 1)
        # calc conditional entropy: H(X|Y)=-sum_(x,y) p(x,y)log(p(x,y)/p(y))
        hy = tf.reduce_sum(tf.math.xlogy(py, py), 1)
        hxy = tf.reduce_sum(tf.math.xlogy(pxy, pxy), [1, 2])
        cond_entropy = hy - hxy
        mean_cond_entropy = tf.reduce_mean(cond_entropy)
        return mean_cond_entropy

    def ecdf(self, maps):
        # calculate the CDF of p
        p = tf.reduce_sum(maps, 1) / self.n_pixels  # shape=(batch_size,bin_bum)
        return tf.cumsum(p, 1)

    def emd_loss(self, maps, maps_hat):
        ecdf_p = self.ecdf(maps)  # shape=(batch_size, bin_bum)
        ecdf_p_hat = self.ecdf(maps_hat)  # shape=(batch_size, bin_bum)
        emd = tf.reduce_mean(tf.pow(tf.abs(ecdf_p - ecdf_p_hat), 2), axis=-1)  # shape=(batch_size,1)
        emd = tf.pow(emd, 1 / 2)
        return tf.reduce_mean(emd)  # shape=0

    def calc_hist_loss_tar_out(self):
        return self.emd_loss(self.maps_tar, self.maps_out)

    def calc_cond_entropy_loss_tar_out(self):
        return self.calc_cond_entropy_loss(self.maps_tar, self.maps_out)

    def calc_relative_mi_tar_out(self):
        return self.calc_relative_mi(self.maps_tar, self.maps_out)


class HistogramLayersCT(HistogramLayers):
    """ Used for Color Transfer
    EMD(TAR, OUT), MI(SRC, OUT)
    """

    def __init__(self,  out, tar, src, args):
        super().__init__(out, tar, args)
        self.maps_src = self.calc_activation_maps(src)

    @tf.function
    def calc_cond_entropy_loss_src_out(self):
        return self.calc_cond_entropy_loss(self.maps_src, self.maps_out)