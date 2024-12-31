import argparse

import torch.random
from torch import optim, nn
from models.model import *
from torch.utils.data import DataLoader
from utils.dataloader import LoadDataset
from utils.transforms import *
from utils import tools
from utils.mix_augmentation import *
from loss.loss import *
from tqdm import tqdm
import sys
import warnings

warnings.filterwarnings('ignore')

def init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        if m.bias is not None:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        else:
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')


"""
"lan" in variables represents Label Alignment Net (CSACLC)
"wrn" in variables represents Weather Restoration Net (De-wea)
"""
def train(opt, device, train_loader, val_loader, csaclc_model, dewea_model, optimizer_lan, optimizer_wrn):
    L_lanPIX, L_lanSSIM = LSmoothMae(), LMSSSIM()
    L_lanROBUST = LRainrobust(
        batch_size=opt.batch_size,
        n_views=2,
        device=torch.device(device),
        temperature=0.25
    )
    L_wrnPIX, L_wrnSSIM, L_wrnDISTILL =\
        LSmoothMae(), LMSSSIM(), LSmoothMae()
    L_wrnROBUST = LRainrobust(
        batch_size=opt.batch_size,
        n_views=2,
        device=torch.device(device),
        temperature=0.25
    )

    for epoch in range(1, opt.epochs + 1):
        print('\n>> ===================== epoch {num} ===================== <<'.format(num=epoch))
        csaclc_model.train()
        dewea_model.train()
        lan_tr_loss, wrn_tr_loss = [], []
        for batch_idx, (img, img_coll, _) in enumerate(tqdm(train_loader, desc='Training', file=sys.stdout)):
            img = img.to(device)
            img_coll = [ts.to(device) for ts in img_coll]
            optimizer_lan.zero_grad()
            optimizer_wrn.zero_grad()

            lan_out, lan_x_clean_proj, lan_x_fea_g = csaclc_model(img, img_coll)
            wrn_out, wrn_x_clean_proj, wrn_x_fea = dewea_model(img, img_coll[-1])

            """ LANet loss"""
            l_langt = (L_lanPIX(lan_out, img_coll[-1]) + L_lanSSIM(lan_out, img_coll[-1])) * .8
            l_lanin = (L_wrnPIX(lan_out, img) + L_wrnSSIM(lan_out, img)) * .2
            l_lanrobust = L_lanROBUST(lan_x_clean_proj) * 0.1
            loss_lan = l_langt + l_lanin + l_lanrobust
            lan_tr_loss.append(loss_lan.item())

            """ WRNet loss"""
            l_wrngt = (L_wrnPIX(wrn_out, img_coll[-1]) + L_wrnSSIM(wrn_out, img_coll[-1])) * .8
            l_wrnlan = (L_wrnPIX(wrn_out, lan_out.detach()) + L_wrnSSIM(wrn_out, lan_out.detach())) * .2
            l_wrnrain = L_wrnROBUST(wrn_x_clean_proj) * 0.1
            l_wrndistill = L_wrnDISTILL(wrn_x_fea, lan_x_fea_g.detach()) * 0.01
            loss_wrn = l_wrngt + l_wrnlan + l_wrnrain + l_wrndistill
            wrn_tr_loss.append(loss_wrn.item())

            loss_lan.backward()
            torch.nn.utils.clip_grad_norm(csaclc_model.parameters(), opt.grad_clip_norm)
            optimizer_lan.step()

            loss_wrn.backward()
            torch.nn.utils.clip_grad_norm(dewea_model.parameters(), opt.grad_clip_norm)
            optimizer_wrn.step()

        print('LANet train loss:', sum(lan_tr_loss) / len(lan_tr_loss))
        print('WRNet train loss:', sum(wrn_tr_loss) / len(wrn_tr_loss))

        if ((epoch) % opt.save_period) == 0:# and epoch >=7:
            checkpoint_cl = {
                'net': csaclc_model.state_dict(),
                'optimizer': optimizer_lan.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint_cl,
                       os.path.join(opt.ckpt_folder, 'epoch_{num}_lanmodel.pth'.format(num=str(epoch))))
            checkpoint_out = {
                'net': dewea_model.state_dict(),
                'optimizer': optimizer_wrn.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint_out,
                       os.path.join(opt.ckpt_folder, 'epoch_{num}_wrnmodel.pth'.format(num=str(epoch))))

        if ((epoch) % opt.val_period) == 0:
            csaclc_model.eval()
            dewea_model.eval()
            lan_val_loss, wrn_val_loss = [], []
            with torch.no_grad():
                for _, (img, img_aid, img_path) in enumerate(tqdm(val_loader, desc='Validating', file=sys.stdout)):
                    img = img.to(device)
                    img_coll = [ts.to(device) for ts in img_coll]

                    lan_out, lan_x_clean_proj, lan_x_fea_g = csaclc_model(img, img_coll)
                    wrn_out, wrn_x_clean_proj, wrn_x_fea = dewea_model(img, img_coll[-1])

                    """ LANet loss"""
                    l_langt = (L_lanPIX(lan_out, img_coll[-1]) + L_lanSSIM(lan_out, img_coll[-1])) * .3
                    l_lanin = (L_wrnPIX(lan_out, img) + L_wrnSSIM(lan_out, img)) * .7
                    l_lanrobust = L_lanROBUST(lan_x_clean_proj) * 0.1
                    loss_lan = l_langt + l_lanin + l_lanrobust
                    lan_val_loss.append(loss_lan.item())

                    """ WRNet loss"""
                    l_wrngt = (L_wrnPIX(wrn_out, img_coll[-1]) + L_wrnSSIM(wrn_out, img_coll[-1])) * .8
                    l_wrnlan = (L_wrnPIX(wrn_out, lan_out.detach()) + L_wrnSSIM(wrn_out, lan_out.detach())) * .2
                    l_wrnrain = L_wrnROBUST(wrn_x_clean_proj) * 0.1
                    l_wrndistill = L_wrnDISTILL(wrn_x_fea, lan_x_fea_g.detach()) * 0.01
                    loss_wrn = l_wrngt + l_wrnlan + l_wrnrain + l_wrndistill
                    wrn_val_loss.append(loss_wrn.item())

            print('ALNet val loss:', sum(lan_val_loss) / len(lan_val_loss))
            print('WRNet val loss:', sum(wrn_val_loss) / len(wrn_val_loss))


def main(opt):
    # check input dir
    assert os.path.exists(opt.train_data), 'train_data folder {dir} does not exist'.format(dir=opt.train_data)
    assert os.path.exists(opt.val_data), 'val_data folder {dir} does not exist'.format(dir=opt.val_data)
    if not os.path.exists(opt.ckpt_folder):
        os.makedirs(opt.ckpt_folder)
        os.makedirs(os.path.join(opt.ckpt_folder, 'step_save'))
    device_id = 'cuda:' + opt.device
    device = torch.device(device_id if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')

    # load datasets
    train_transform = Compose([RandomResizedCrop([opt.imgsz[0], opt.imgsz[1]]),
                               RandomApply([RandomRotation(degrees=20, expand=False)
                                            ], p=0.66),
                               RandomSelectFlip(p=0.33),
                               ToTensor()])
    train_set = LoadDataset(opt.train_data, train_transform, mode='train')
    train_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers,
                              batch_size=opt.batch_size, shuffle=True, drop_last=True)

    val_transform = Compose([RandomResizedCrop([opt.imgsz[0], opt.imgsz[1]]),
                               RandomApply([RandomRotation(degrees=20, expand=False)
                                            ], p=0.66),
                               RandomSelectFlip(p=0.33),
                               ToTensor()])
    val_set = LoadDataset(opt.val_data, val_transform, mode='train')
    val_loader = DataLoader(dataset=val_set, num_workers=opt.num_workers,
                              batch_size=opt.batch_size, shuffle=True, drop_last=True)

    print('Num of train_set set: {num}'.format(num=len(train_set)))
    print('Num of val_set set: {num}'.format(num=len(val_set)))

    csaclc_model = CSACLC().to(device)
    dewea_model = Dewea_train().to(device)

    optimizer_lan = optim.Adam(csaclc_model.parameters(), lr=opt.lr,
                              betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weight_decay)
    optimizer_wrn = optim.Adam(dewea_model.parameters(), lr=opt.lr,
                               betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weight_decay)

    # start training
    train(opt, device, train_loader, val_loader, csaclc_model, dewea_model, optimizer_lan, optimizer_wrn)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str,
                        default='path to train set',
                        help='train_set set path')
    parser.add_argument('--val_data', type=str,
                        default='path to val set',
                        help='val_set set path')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
    parser.add_argument('--num_workers', type=int, default=6, help='dataloader workers')
    parser.add_argument('--imgsz', type=int, default=[256, 256], help='input image size')
    parser.add_argument('--epochs', type=int, default=50, help='total epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1, help='grad_clip_norm')
    parser.add_argument('--device', default='0', help='use cuda device; 0, 1, 2 or cpu')
    parser.add_argument('--save_period', type=int, default=1, help='save checkpoint every x epoch')
    parser.add_argument('--val_period', type=int, default=1, help='perform validation every x epoch')
    parser.add_argument('--ckpt_folder', type=str, default='path to output folder', help='location for saving ckpts')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)