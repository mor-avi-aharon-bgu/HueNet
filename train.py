import tensorflow as tf
import os
import datetime
import time

import pathlib
import random

import pylib as py
from prepare_dataset import load_image, load_single_image, save_images, shuffle_zip
from models import Generator, Discriminator
from loss import generator_loss, discriminator_loss, generator_loss_color_transfer
import atexit
import config

# parameters
parser = config.options()
config.define_task_default_params(parser)
args = parser.parse_args()

# output_dir
output_dir = py.join(args.output_dir, args.dataset)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# Input Pipeline
PATH = args.dataroot + '/' + args.dataset + '/'
paired = False if args.task == 'color_transfer' else True

if paired:
    # create paired dataset
    if args.dataset == 'summer2winter_yosemite':
        train_ds = tf.data.Dataset.list_files(PATH + 'train*/*.jpg')
        test_ds = tf.data.Dataset.list_files(PATH + 'test*/*.jpg')
    else:
        train_ds = tf.data.Dataset.list_files(PATH + 'train/*.jpg')
        test_ds = tf.data.Dataset.list_files(PATH + 'val/*.jpg')

    train_ds = train_ds.map(lambda x: load_image(image_file=x, train=True, args=args))
    train_ds = train_ds.shuffle(args.buffer_size)
    train_ds = train_ds.batch(args.batch_size)
    test_ds = test_ds.map(lambda x: load_image(image_file=x, train=False, args=args))
    test_ds = test_ds.batch(args.batch_size)

else:
    # create unpaired dataset
    data_root = pathlib.Path(PATH)
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]

    # shuffle paths
    random.seed(1)
    random.shuffle(all_image_paths)

    # split to train and validation
    TRAIN_PER = 0.9
    train_size = round(len(all_image_paths) * TRAIN_PER)
    train_paths = all_image_paths[0:train_size]
    val_paths = all_image_paths[train_size:]

    # create dataset
    train_A = tf.data.Dataset.from_tensor_slices(train_paths).map(
        lambda x: load_single_image(image_file=x, hist=False, train=True, args=args))
    train_B = tf.data.Dataset.from_tensor_slices(train_paths).map(
        lambda x: load_single_image(image_file=x, hist=True, train=True, args=args))
    test_A = tf.data.Dataset.from_tensor_slices(val_paths).map(
        lambda x: load_single_image(image_file=x, hist=False, train=False, args=args))
    test_B = tf.data.Dataset.from_tensor_slices(val_paths).map(
        lambda x: load_single_image(image_file=x, hist=True, train=False, args=args))

# Define the models
generator = Generator(args=args)
discriminator = Discriminator(args=args, conditional=paired)

# Define the Optimizers and Checkpoint-saver
generator_optimizer = tf.keras.optimizers.Adam(args.gen_lr, beta_1=args.gen_beta_1)
discriminator_optimizer = tf.keras.optimizers.Adam(args.dis_lr, beta_1=args.dis_beta_1)

checkpoint_dir = py.join(output_dir, 'checkpoints')
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
# restore checkpoints
if args.output_pre_dir:
    # restoring the latest checkpoint in checkpoint_dir
    output_pre_dir = py.join(args.output_pre_dir, args.dataset)
    checkpoint_pre_dir = py.join(output_pre_dir, 'checkpoints')
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_pre_dir))
    print('restore checkpoint: ' + tf.train.latest_checkpoint(checkpoint_pre_dir))
else:
    print('No checkpoint to restore ...')


def exit_handler():
    print('Save checkpoint before exit...')
    checkpoint.save(file_prefix=checkpoint_prefix)


atexit.register(exit_handler)

# sample
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

# logs
log_dir = py.join(output_dir, 'logs/')
summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


# Train
@tf.function
def train_step(input_image, target, target_hists):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator([input_image, target_hists[0], target_hists[1], target_hists[2]], training=True)

        if paired:
            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator([input_image, gen_output], training=True)
            gen_total_loss, gen_gan_loss, gen_mi_loss, gen_hist_loss = generator_loss_color_transfer(
                disc_generated_output, gen_output, input_image, target, args)
        else:
            disc_real_output = discriminator(input_image, training=True)
            disc_generated_output = discriminator(gen_output, training=True)
            gen_total_loss, gen_gan_loss, gen_mi_loss, gen_hist_loss = generator_loss(disc_generated_output, gen_output,
                                                                                      target, args)

        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    return {'gen_total_loss': gen_total_loss,
            'gen_gan_loss': gen_gan_loss,
            'gen_mi_loss': gen_mi_loss,
            'gen_hist_loss': gen_hist_loss,
            'disc_loss': disc_loss}


def fit():
    total_loss_dict = {'gen_total_loss': 0, 'gen_gan_loss': 0, 'gen_mi_loss': 0, 'gen_hist_loss': 0, 'disc_loss': 0}

    for epoch in range(args.epochs):
        start = time.time()
        global train_ds, test_ds

        if not paired:
            train_ds = shuffle_zip(train_A, train_B, args)
            test_ds = shuffle_zip(test_A, test_B, args)

        for example_input, (example_target, example_target_hists) in test_ds.take(1):
            save_images(generator, example_input, example_target, example_target_hists,
                        py.join(sample_dir, 'epoch-%09d.jpg' % epoch), args)

        print("Epoch: ", epoch)

        # Train
        total_loss_dict = dict.fromkeys(total_loss_dict, 0)  # initialize total loss with zero

        for n, (input_image, (target, target_hists)) in train_ds.enumerate():

            loss_dict = train_step(input_image, target, target_hists)

            for loss_name in total_loss_dict:
                total_loss_dict[loss_name] += loss_dict[loss_name]

            if (n + 1) % 100 == 0 or n == 0:
                print(n.numpy() + 1, end=' ')
                for loss_name in loss_dict:
                    print(loss_name, ':', loss_dict[loss_name].numpy(), end=' ')
                print()

        # write logs
        with summary_writer.as_default():
            for loss_name in total_loss_dict:
                tf.summary.scalar(loss_name, total_loss_dict[loss_name] / float(n + 1), step=epoch)
            summary_writer.flush()  # Mor: try to fix tensor-board graphs

        # saving (checkpoint) the model every args.cp_freq epochs
        if (epoch + 1) % args.cp_freq == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            print("save checkpoint ...")

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))
    checkpoint.save(file_prefix=checkpoint_prefix)
    print("save checkpoint ...")


fit()
