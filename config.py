import argparse


def options():
    parser = argparse.ArgumentParser()

    # basic parameters
    parser.add_argument('--dataroot', required=True, help='path to data directory (should have subfolder with the dataset name)')
    parser.add_argument('--task', default='edges2photos', choices=['edges2photos', 'color_transfer', 'colorization'])
    parser.add_argument('--dataset', choices=['edges2shoes', 'edges2handbags', '102flowers', 'summer2winter_yosemite'])
    parser.add_argument('--output_dir', default='output', help='models, samples and logs are saved here')
    parser.add_argument('--output_pre_dir', default='', help='if specified load checkpoint from this directory')

    # model parameters
    parser.add_argument('--img_h', type=int, default=256, help='crop image to this image height')
    parser.add_argument('--img_w', type=int, default=256, help='crop image to this image width')
    parser.add_argument('--img_c', type=int, default=3, help='# of input image channels')
    parser.add_argument('--img_out_c', type=int, default=3, help='# of output image channels')

    # histogram layers parameters
    parser.add_argument('--bin_num', type=int, default=256, help='histogram layers - number of bins')
    parser.add_argument('--kernel_width_ratio', type=float, default=2.5, help='histogram layers - scale kernel width')

    # optimizer parameters
    parser.add_argument('--gen_lr', type=float, default=2e-4)
    parser.add_argument('--gen_beta_1', type=float, default=0.5)
    parser.add_argument('--dis_lr', type=float, default=2e-4)
    parser.add_argument('--dis_beta_1', type=float, default=0.5)

    # data prepare parameters
    parser.add_argument('--min_val', type=float, default=-1.0, help="normalize image values to this min")
    parser.add_argument('--max_val', type=float, default=1.0, help="normalize image values to this max")
    parser.add_argument('--yuv', type=bool, default=True, help="convert images to YUV colorspace")

    # training params
    parser.add_argument('--batch_size', type=int, help='input batch size')
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--cp_freq', type=int, help='checkpoint frequency - save every # epochs')
    parser.add_argument('--buffer_size', type=int, help='size of shuffle buffer')

    # data prepare params
    parser.add_argument('--a2b', type=bool, help='image translation direction AtoB or BtoA')
    parser.add_argument('--jitter', type=bool, help='random jitter for data augmentation')

    # loss params
    parser.add_argument('--gan_loss_weight', type=float)
    parser.add_argument('--mi_loss_weight', type=float)
    parser.add_argument('--hist_loss_weight', type=float)

    return parser


def define_task_default_params(parser):
    args = parser.parse_args()

    if args.task == 'colorization':
        parser.set_defaults(dataset='summer2winter_yosemite', batch_size=4, epochs=200, cp_freq=20, buffer_size=400,
                            a2b=True, jitter=True, gan_loss_weight=1, mi_loss_weight=1, hist_loss_weight=20)

    elif args.task == 'color_transfer':
        parser.set_defaults(dataset='102flowers', batch_size=4, epochs=100, cp_freq=10, buffer_size=400, a2b=False,
                            jitter=True, gan_loss_weight=1, mi_loss_weight=1, hist_loss_weight=100)

    elif args.task == 'edges2photos':
        parser.set_defaults(dataset='edges2shoes', batch_size=4, epochs=15, cp_freq=5, buffer_size=400, a2b=False,
                            jitter=False, gan_loss_weight=1, mi_loss_weight=1, hist_loss_weight=100)
