import tensorflow as tf
from histogram_layers import HistogramLayers, HistogramLayersCT

bc_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# generator loss mi
def generator_loss_color_transfer(disc_generated_output, gen_output, inp, target, args):
    gan_loss = bc_loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # histograms instances
    hist_1 = HistogramLayersCT(out=gen_output[..., 0], tar=target[..., 0], src=inp[..., 0], args=args)
    hist_2 = HistogramLayersCT(out=gen_output[..., 1], tar=target[..., 1], src=inp[..., 1], args=args)
    hist_3 = HistogramLayersCT(out=gen_output[..., 2], tar=target[..., 2], src=inp[..., 2], args=args)

    # mi loss
    mi_loss_1 = hist_1.calc_cond_entropy_loss_src_out()
    mi_loss_2 = hist_2.calc_cond_entropy_loss_src_out()
    mi_loss_3 = hist_3.calc_cond_entropy_loss_src_out()

    mi_loss = (mi_loss_1 + mi_loss_2 + mi_loss_3) / 3

    # hist loss
    hist_loss_1 = hist_1.calc_hist_loss_tar_out()
    hist_loss_2 = hist_2.calc_hist_loss_tar_out()
    hist_loss_3 = hist_3.calc_hist_loss_tar_out()

    hist_loss = (hist_loss_1 + hist_loss_2 + hist_loss_3) / 3

    total_gen_loss = (args.gan_loss_weight * gan_loss) + (args.mi_loss_weight * mi_loss) + \
                     (args.hist_loss_weight * hist_loss)

    return total_gen_loss, gan_loss, mi_loss, hist_loss


def generator_loss(disc_generated_output, gen_output, target, args):
    gan_loss = bc_loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # histograms instances
    hist_1 = HistogramLayers(out=gen_output[..., 0], tar=target[..., 0], args=args)
    hist_2 = HistogramLayers(out=gen_output[..., 1], tar=target[..., 1], args=args)
    hist_3 = HistogramLayers(out=gen_output[..., 2], tar=target[..., 2], args=args)

    # mi loss
    mi_loss_1 = hist_1.calc_cond_entropy_loss_tar_out()
    mi_loss_2 = hist_2.calc_cond_entropy_loss_tar_out()
    mi_loss_3 = hist_3.calc_cond_entropy_loss_tar_out()

    mi_loss = (mi_loss_1 + mi_loss_2 + mi_loss_3) / 3

    # hist loss
    hist_loss_1 = hist_1.calc_hist_loss_tar_out()
    hist_loss_2 = hist_2.calc_hist_loss_tar_out()
    hist_loss_3 = hist_3.calc_hist_loss_tar_out()

    hist_loss = (hist_loss_1 + hist_loss_2 + hist_loss_3) / 3

    total_gen_loss = (args.gan_loss_weight * gan_loss) + (args.mi_loss_weight * mi_loss) + (
                args.hist_loss_weight * hist_loss)

    return total_gen_loss, gan_loss, mi_loss, hist_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = bc_loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = bc_loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss
