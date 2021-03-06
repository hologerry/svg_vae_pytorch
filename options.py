import argparse
# TODO: add help for the parameters


def get_parser_basic():
    parser = argparse.ArgumentParser()
    # TODO: basic parameters training related
    parser.add_argument('--model_name', type=str, default='image_vae', choices=['image_vae', 'svg_decoder'],
                        help='current model_name')
    parser.add_argument('--initializer', type=str, default='uniform_unit_scaling',
                        choices=['uniform', 'orthogonal', 'uniform_unit_scaling'],
                        help='image vae initializer type')
    parser.add_argument('--initializer_gain', type=float, default=1.0, help='image vae initializer initializer gain')
    parser.add_argument('--bottleneck_bits', type=int, default=32, help='image vae number of bottleneck bits')
    parser.add_argument('--kl_beta', type=float, default=1.0, help='image vae kl loss beta')
    parser.add_argument('--free_bits_div', type=int, default=4, help='image vae free bits div, not used')
    parser.add_argument('--free_bits', type=float, default=0.15, help='image vae free bits')
    parser.add_argument('--num_categories', type=int, default=52, help='number of glyphs, original is 62')
    parser.add_argument('--in_channel', type=int, default=1, help='input image channel')
    parser.add_argument('--out_channel', type=int, default=1, help='output image channel')
    # data related
    parser.add_argument('--max_seq_len', type=int, default=51, help='maximum length of sequence')
    parser.add_argument('--seq_feature_dim', type=int, default=10,
                        help='feature dim (like vocab size) of one step of sequence feature')
    # experiment related
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--experiment_name', type=str, default='experiment_00_10')
    parser.add_argument('--data_root', type=str, default='svg_vae_data/glyph_pkl_dataset_10')
    parser.add_argument('--ckpt_freq', type=int, default=2, help='save checkpoint frequency of epoch')
    parser.add_argument('--sample_freq', type=int, default=100, help='sample train output of steps')
    parser.add_argument('--val_freq', type=int, default=1000, help='sample validate output of steps')
    parser.add_argument('--beta1', type=float, default=0.85, help='beta1 of Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.997, help='beta2 of Adam optimizer')
    parser.add_argument('--eps', type=float, default=1e-6, help='Adam epsilon')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='image vae weight decay')
    parser.add_argument('--tboard', type=bool, default=True, help='whether use tensorboard to visulize loss')
    # TODO: weight decay
    return parser


def get_parser_image_vae():
    parser = get_parser_basic()
    # experiment
    parser.add_argument('--batch_size', type=int, default=64, help='image vae batch_size')
    parser.add_argument('--init_epoch', type=int, default=0, help='init epoch')
    parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    # model_name
    parser.add_argument('--hidden_size', type=int, default=32, help='image vae hidden_size, not used')
    parser.add_argument('--base_depth', type=int, default=32, help='image vae conv layer base depth')
    # problem related not clear
    parser.add_argument('--absolute', type=bool, default=False, help='')
    parser.add_argument('--just_render', type=bool, default=True, help='')
    parser.add_argument('--plus_render', type=bool, default=False, help='')

    return parser


def get_parser_svg_decoder():
    parser = get_parser_basic()
    parser.add_argument('--batch_size', type=int, default=128, help='svg decoder batch size')
    parser.add_argument('--init_epoch', type=int, default=0, help='init epoch')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--hidden_size', type=int, default=256, help='svg decoder hidden size')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='svg decoder number of hidden layers')
    parser.add_argument('--force_full_predict', type=bool, default=True, help='')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    # parser.add_argument('--dropout', type=float, default=0.5, help='svg decoder dropout layer probability')
    parser.add_argument('--learning_rate_warmup_steps', type=int, default=1e5, help='')
    parser.add_argument('--vocab_size', type=int, default=None, help='')
    # loss params
    parser.add_argument('--soft_k', type=int, default=10, help='')
    parser.add_argument('--mdn_k', type=int, default=1, help='')
    # LSTM
    parser.add_argument('--rec_dropout', type=int, default=0.3, help='LayerNormLSTMCelll, recurrent dropout')
    parser.add_argument('--ff_dropout', type=bool, default=True, help='input dropout of LSTM')
    # Decode architecture
    parser.add_argument('--twice_decoder', type=bool, default=False, help='')
    parser.add_argument('--sg_bottleneck', type=bool, default=True,
                        help='stop gradient bottleneck, if True, fix the vae train the decoder only')
    parser.add_argument('--condition_on_sln', type=bool, default=False, help='')
    parser.add_argument('--use_cls', type=bool, default=True, help='')
    # MDNl loss
    parser.add_argument('--num_mixture', type=int, default=50, help='')
    parser.add_argument('--mix_temperature', type=float, default=0.0001, help='')
    parser.add_argument('--gauss_temperature', type=float, default=0.0001, help='')
    parser.add_argument('--dont_reduce_loss', type=bool, default=False, help='')
    # VAE hparameters (to load image encoder)
    parser.add_argument('--vae_ckpt_path', type=str, default='experiments/experiment_00_10_image_vae/checkpoints/image_vae_200.pth')
    # parser.add_argument('--vae_data_dir', type=str, default='')
    # problem related not clear
    parser.add_argument('--absolute', type=bool, default=False, help='')
    parser.add_argument('--just_render', type=bool, default=False, help='')
    parser.add_argument('--plus_render', type=bool, default=False, help='')

    return parser
