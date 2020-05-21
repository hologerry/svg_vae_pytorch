import argparse
import multiprocessing as mp
import os
import pickle

import data_utils.svg_utils as svg_utils


'''
{'uni': int64,  # unicode value of this glyph
'width': int64,  # width of this glyph's viewport (provided by fontforge)
'vwidth': int64,  # vertical width of this glyph's viewport
'sfd': binary/str,  # glyph, converted to .sfd format, with a single SplineSet
'id': binary/str,  # id of this glyph
'binary_fp': binary/str}  # font identifier (provided in glyphazzn_urls.txt)
'''


def create_db(opts):
    all_font_ids = sorted(os.listdir(os.path.join(opts.sfd_path, opts.split)))
    num_fonts = len(all_font_ids)
    fonts_per_process = num_fonts // opts.num_processes
    char_num = 52

    def process(process_id):
        cur_process_processed_font_glyphs = []
        cur_process_log_file = open(os.path.join(opts.log_dir, f'log_{process_id}.txt'), 'w')
        cur_process_pkl_file = open(os.path.join(opts.output_path, opts.split, f'{opts.split}_{process_id:04d}-{opts.num_processes+1:04d}.pkl'), 'wb')
        for i in range(process_id * fonts_per_process, (process_id + 1) * fonts_per_process):
            if i >= num_fonts:
                break
            font_id = all_font_ids[i]
            cur_font_sfd_dir = os.path.join(opts.sfd_path, opts.split, font_id)
            for char_id in range(char_num):
                char_desp_f = open(os.path.join(cur_font_sfd_dir, '{}_{:02d}.txt'.format(font_id, char_id)), 'r')
                char_desp = char_desp_f.readlines()
                char_desp_f.close()
                sfd_f = open(os.path.join(cur_font_sfd_dir, '{}_{:02d}.sfd'.format(font_id, char_id)), 'rb')
                sfd = sfd_f.read()
                sfd_f.close()

                uni = int(char_desp[0].strip())
                width = int(char_desp[1].strip())
                vwidth = int(char_desp[2].strip())
                char_idx = char_desp[3].strip()
                font_idx = char_desp[4].strip()

                cur_glyph = {}
                cur_glyph['uni'] = uni
                cur_glyph['width'] = width
                cur_glyph['vwidth'] = vwidth
                cur_glyph['sfd'] = sfd
                cur_glyph['id'] = char_idx
                cur_glyph['binary_fp'] = font_idx

                if not svg_utils.is_valid_glyph(cur_glyph):
                    msg = f"font {font_idx}, char {char_idx} is not a valid glyph\n"
                    cur_process_log_file.write(msg)
                    print(msg)
                pathunibfp = svg_utils.convert_to_path(cur_glyph)
                if not svg_utils.is_valid_path(pathunibfp):
                    msg = f"font {font_idx}, char {char_idx}'s sfd is not a valid path\n"
                    cur_process_log_file.write(msg)

                example = svg_utils.create_example(pathunibfp)
                cur_process_processed_font_glyphs.append(example)

        pickle.dump(cur_process_processed_font_glyphs, cur_process_pkl_file)
        cur_process_pkl_file.close()

    processes = [mp.Process(target=process, args=(pid)) for pid in range(opts.process_nums + 1)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()


def combine_perprocess_pkl_db(opts):
    all_glyphs = []
    all_glyphs_pkl_file = open(os.path.join(opts.output_path, opts.split, f'all.pkl'), 'wb')
    for process_id in range(opts.num_processes + 1):
        cur_process_pkl_file = open(os.path.join(opts.output_path, opts.split, f'{opts.split}_{process_id:04d}-{opts.num_processes+1:04d}.pkl'), 'rb')
        cur_process_glyphs = pickle.load(cur_process_pkl_file)
        all_glyphs += cur_process_glyphs
    pickle.dump(all_glyphs, all_glyphs_pkl_file)
    return len(all_glyphs)


def cal_mean_stddev(opts):
    # TODO: calculate mean_stddev of the seqence length
    pass


def main():
    parser = argparse.ArgumentParser(description="LMDB creation")
    parser.add_argument('--sfd_path', type=str, default='svg_vae_data/sfd_font_glyphs_mp')
    parser.add_argument("--output_path", type=str, default='svg_vae_data/glyph_pkl_dataset',
                        help="Path to write the database to")
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--log_dir", type=str, default='svg_vae_data/create_pkl_log/')

    opts = parser.parse_args()
    assert os.path.exists(opts.sfd_path), "specified sfd glyphs path does not exist"
    split_path = os.path.join(opts.output_path, opts.split)

    if not os.path.exists(split_path):
        os.makedirs(split_path)

    if not os.path.exists(opts.log_dir):
        os.makedirs(opts.log_dir)
    opts.num_processes = mp.cpu_count() - 2

    create_db(opts)

    number_saved_glyphs = combine_perprocess_pkl_db(opts)
    print(number_saved_glyphs)


if __name__ == "__main__":
    main()
