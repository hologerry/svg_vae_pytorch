import argparse


def get_parser_basic():
    parser = argparse.ArgumentParser()
    # TODO: basic parameters
    return parser


def get_parser_image_vae():
    parser = get_parser_basic()
    parser.add_argument('--batch_size', type=int, default=64, help='image vae batch_size')
    parser.add_argument('--hidden_size', type=int, default=32, help='image vae hidden_size')
    parser.add_argument('--initializer', type=str, default='uniform_unit_scaling', help='image vae initializer type')
    parser.add_argument('--initializer_gain', type=float, default=1.0, help='image vae initializer initializer gain')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='image vae weight decay')
    parser.add_argument('--base_depth', type=int, default=32, help='image vae conv layer base depth')
    parser.add_argument('--bottleneck_bits', type=int, default=32, help='image vae number of bottleneck bits')
    parser.add_argument('--kl_beta', type=int, default=300, help='')
    parser.add_argument('--free_bits_div', type=int, default=4, help='')
    parser.add_argument('--num_categories', type=int, default=52, help='number of glyphs, original is 62')
    parser.add_argument('--absolute', type=bool, default=False, help='')
    parser.add_argument('--just_render', type=bool, default=True, help='')
    parser.add_argument('--plus_render', type=bool, default=False, help='')

    return parser
