import data_utils.svg_utils as svg_utils


char_desp_f = open('svg_vae_data/sfd_font_glyphs_mp/train/002360/002360-44.txt', 'r')
char_desp = char_desp_f.readlines()
char_desp_f.close()
sfd_f = open('svg_vae_data/sfd_font_glyphs_mp/train/002360/002360-44.sfd', 'r')
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
    print(msg)
    # use the font whose all glyphs are valid
pathunibfp = svg_utils.convert_to_path(cur_glyph)
if not svg_utils.is_valid_path(pathunibfp):
    msg = f"font {font_idx}, char {char_idx}'s sfd is not a valid path\n"

example = svg_utils.create_example(pathunibfp)
