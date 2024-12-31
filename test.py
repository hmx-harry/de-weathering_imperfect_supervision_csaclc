import argparse
from PIL import Image
import os.path
from tabulate import tabulate
from models.model import *
from glob import glob
import numpy as np
import warnings
from natsort import natsorted
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
warnings.filterwarnings('ignore')

def prepros(im, scale=8):
    im = np.array(im, dtype=np.float32)
    im *= 1 / 255
    height, width = im.shape[:2]
    im = im[:height - height % scale, :width - width % scale, :]
    return im

def np2tsr(im):
    im = torch.from_numpy(im).permute((2, 0, 1))
    im = torch.unsqueeze(im, 0).cuda()
    return im

def custom_print(*args, **kwargs):
    global print_content
    output = " ".join(map(str, args))
    print_content.append(output)
    print(output)

def main(opt):
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    device_id = 'cuda:' + opt.device
    device = torch.device(device_id if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')

    model = Dewea_test().to(device)
    print('Loading weights:', opt.weights)
    checkpoint = torch.load(opt.weights)
    model.load_state_dict(checkpoint['net'])

    model.eval()

    scene_paths = natsorted(glob(os.path.join(opt.test_data, '*')))
    scene_val_dict = {}
    for scene_path in scene_paths:
        scene_name = scene_path.split('/')[-1]
        print(scene_name)
        clean_img_path = glob(os.path.join(scene_path, '*C-000.*'))
        if len(clean_img_path) == 0:
            clean_img_path = glob(os.path.join(scene_path, 'gt.*'))
        clean_img_path = clean_img_path[0]

        input_img_paths = natsorted(glob(scene_path + '/*-R-*.*'))
        if len(input_img_paths) == 0:
            input_img_paths = natsorted(glob(scene_path + '/degraded_*.*'))

        scene_PSNR_in, scene_SSIM_in = 0, 0
        scene_PSNR_out, scene_SSIM_out = 0, 0

        for i in range(len(input_img_paths)):
            filename = os.path.basename(input_img_paths[i])
            img = Image.open(input_img_paths[i])
            gt_img = Image.open(clean_img_path)

            img = prepros(img)
            input = np2tsr(img) * 2 - 1
            gt_img = prepros(gt_img)
            output = (model(input)[0] * 0.5 + 0.5).squeeze().permute((1, 2, 0))
            output = output.detach().cpu().numpy()

            scene_PSNR_in += psnr(gt_img, img)
            scene_SSIM_in += ssim(gt_img, img, multichannel=True)
            scene_PSNR_out += psnr(gt_img, output)
            scene_SSIM_out += ssim(gt_img, output, multichannel=True)

            draw = output
            draw = Image.fromarray((draw * 255).astype(np.uint8))
            if not os.path.exists(f"{opt.save_path}/{scene_name}/"):
                os.makedirs(f"{opt.save_path}/{scene_name}/")
            draw.save(f"{opt.save_path}/{scene_name}/{filename}.png")

        scene_PSNR_avg_in = round(scene_PSNR_in / len(input_img_paths), 4)
        scene_SSIM_avg_in = round(scene_SSIM_in / len(input_img_paths), 4)
        scene_PSNR_avg_out = round(scene_PSNR_out / len(input_img_paths), 4)
        scene_SSIM_avg_out = round(scene_SSIM_out / len(input_img_paths), 4)

        scene_val_dict[scene_name] = [scene_PSNR_avg_in, scene_SSIM_avg_in,
                                      scene_PSNR_avg_out, scene_SSIM_avg_out]

        table = [[scene_name, 'PSNR↑', 'SSIM↑'],
                 ['Input', str(scene_PSNR_avg_in), str(scene_SSIM_avg_in)],
                 ['Output', str(scene_PSNR_avg_out), str(scene_SSIM_avg_out)]
                 ]
        custom_print(tabulate(table, headers='firstrow', tablefmt='grid') + '\n')

    tot_PSNR_in, tot_SSIM_in = 0, 0
    tot_PSNR_out, tot_SSIM_out = 0, 0
    for cls in sense_dict.keys():
        cls_PSNR_in, cls_SSIM_in = 0, 0
        cls_PSNR_out, cls_SSIM_out = 0, 0
        for obj_name in sense_dict[cls]:
            cls_PSNR_in += scene_val_dict[obj_name][0]
            cls_SSIM_in += scene_val_dict[obj_name][1]
            cls_PSNR_out += scene_val_dict[obj_name][2]
            cls_SSIM_out += scene_val_dict[obj_name][3]

            tot_PSNR_in += scene_val_dict[obj_name][0]
            tot_SSIM_in += scene_val_dict[obj_name][1]
            tot_PSNR_out += scene_val_dict[obj_name][2]
            tot_SSIM_out += scene_val_dict[obj_name][3]

        group_PSNR_avg_in = round(cls_PSNR_in / len(sense_dict[cls]), 4)
        group_SSIM_avg_in = round(cls_SSIM_in / len(sense_dict[cls]), 4)
        group_PSNR_avg_out = round(cls_PSNR_out / len(sense_dict[cls]), 4)
        group_SSIM_avg_out = round(cls_SSIM_out / len(sense_dict[cls]), 4)

        table = [['Cls:' + cls, 'PSNR↑', 'SSIM↑'],
                 ['Input', str(group_PSNR_avg_in), str(group_SSIM_avg_in)],
                 ['Output', str(group_PSNR_avg_out), str(group_SSIM_avg_out)]
                 ]
        custom_print(tabulate(table, headers='firstrow', tablefmt='grid') + '\n')

    tot_PSNR_avg_in = round(tot_PSNR_in / len(scene_paths), 4)
    tot_SSIM_avg_in = round(tot_SSIM_in / len(scene_paths), 4)
    tot_PSNR_avg_out = round(tot_PSNR_out / len(scene_paths), 4)
    tot_SSIM_avg_out = round(tot_SSIM_out / len(scene_paths), 4)
    table = [['Total', 'PSNR↑', 'SSIM↑'],
             ['Input', str(tot_PSNR_avg_in), str(tot_SSIM_avg_in)],
             ['Output', str(tot_PSNR_avg_out), str(tot_SSIM_avg_out)]
            ]
    custom_print(tabulate(table, headers='firstrow', tablefmt='grid') + '\n')

    with open(os.path.join(opt.save_path, 'metrics.txt'), "w") as file:
        for line in print_content:
            file.write(line + "\n")

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str,
                        default='path to test set',
                        help='train set path')
    parser.add_argument('--weights', type=str,
                        default='path to weights',
                        help='location of model weights')
    parser.add_argument('--save_path',
                        default='path to output folder',
                        help='use cuda device; 0, 1, 2 or cpu')
    parser.add_argument('--device', default='0', help='use cuda device; 0, 1, 2 or cpu')
    return parser.parse_known_args()[0] if known else parser.parse_args()

if __name__ == '__main__':
    sense_dict = {
    'rain': ['123_0', '260_0', '762_0', '921_0', '921_1', '1199_1', '1386_0','1386_1', '1924_0', '3082_2',
             'Gurutto_0-0', 'Winter_Garden_0-1', 'Winter_Garden_0-4',
             'Oinari_1-1','Oinari_0-0', 'M1135_0-0', 'Table_Rock_0-0'],

    'fog': ['47_0', '47_1', '87_0', '161_0', '217_0', '649_0', '1039_0', '1629_0','1853_0', '1853_1',
            '1980_0', '2', '11', '1787', '2856_1'],

    'snow': ['1066_0', '1066_1', '1303', '2582', '2726', '2856', 'Cassiano_0-0',
             'Davison_0-0', 'Flagstaff_0-1', 'Fukui_East_0-6', 'Stick_0-5', 'Teton_0-0', 'Truskavets_0-3']
    }
    print_content = []
    opt = parse_opt()
    main(opt)