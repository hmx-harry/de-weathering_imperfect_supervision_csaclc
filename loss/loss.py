import torch.nn as nn
import torchvision.transforms as transforms
import torch
from torchvision.models import vgg19
import torch.nn.functional as F
from skimage import color
from math import exp
from piq import MultiScaleSSIMLoss
import cv2

class LMae(nn.Module):
    """ L1 loss """
    def __init__(self):
        super(LMae, self).__init__()
        self.mae = torch.nn.L1Loss()

    def forward(self, im1, im2):
        return self.mae(im1, im2)

class LSmoothMae(nn.Module):
    """ Smooth L1 loss """
    def __init__(self):
        super(LSmoothMae, self).__init__()
        self.mae = torch.nn.SmoothL1Loss()

    def forward(self, im1, im2):
        return self.mae(im1, im2)

class LMse(nn.Module):
    """ L2 loss """
    def __init__(self):
        super(LMse, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, im1, im2):
        return self.mse(im1, im2)

class LCol(torch.nn.Module):
    def __init__(self):
        super(LCol, self).__init__()
        self.kernel_size = 5
        self.padding = self.kernel_size // 2
        self.weight = torch.ones(3, 3, self.kernel_size, self.kernel_size) / (self.kernel_size ** 2)

    def forward(self, img1, img2):
        img1_blur = self.blur(img1)
        img2_blur = self.blur(img2)
        img1_lab = self.rgb_to_lab(img1_blur)
        img2_lab = self.rgb_to_lab(img2_blur)
        return F.mse_loss(img1_lab, img2_lab)

    def rgb_to_lab(self, img):
        batch = []
        for idx in range(img.shape[0]):
            img_np = img[idx].mul(255).byte()
            img_np = img_np.detach().cpu().numpy().transpose((1, 2, 0))
            img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            img_lab = torch.from_numpy(img_lab).permute(2, 0, 1).div(255).to(img.device)
            batch.append(img_lab)
        batch = torch.stack(batch, dim=0)
        return batch

    def blur(self, img):
        self.weight = self.weight.to(img.device)
        img = F.conv2d(img, self.weight, padding=self.padding)
        return img


class LMSSSIM(torch.nn.Module):
    def __init__(self):
        super(LMSSSIM, self).__init__()
        self.msssim = MultiScaleSSIMLoss(data_range=1.).to('cuda:0')

    def forward(self, im1, im2):
        im1 = im1 * 0.5 + 0.5
        im2 = im2 * 0.5 + 0.5
        return self.msssim(im1, im2)

class LSc(nn.Module):
    def __init__(self):
        super(LSc, self).__init__()

    def forward(self, im1, im2):
        return 1 - self.ssim_torch(im1, im2)

    def ssim_torch(self, im1, im2, L=1):
        K2 = 0.03
        C2 = (K2 * L) ** 2
        C3 = C2 / 2
        ux = torch.mean(im1)
        uy = torch.mean(im2)
        ox_sq = torch.var(im1)
        oy_sq = torch.var(im2)
        ox = torch.sqrt(ox_sq)
        oy = torch.sqrt(oy_sq)
        oxy = torch.mean((im1 - ux) * (im2 - uy))
        oxoy = ox * oy
        C = (2 * ox * oy + C2) / (ox_sq + oy_sq + C2)
        S = (oxy + C3) / (oxoy + C3)
        return S * C

class LPercep(nn.Module):
    def __init__(self):
        super(LPercep, self).__init__()
        self.vgg19 = vgg19(pretrained=True).features
        #print(self.vgg19)
        self.layers = [3, 8, 13]
        for param in self.vgg19.parameters():
            param.requires_grad = False
        self.flag = False

    def forward(self, im1, im2):
        if not self.flag:
            self.vgg19 = self.vgg19.to(im1.device)
            self.flag = True

        loss = 0.0
        for i, layer in enumerate(self.vgg19):
            im1, im2 = layer(im1), layer(im2)
            if i in self.layers:
                loss += nn.functional.l1_loss(im1, im2)
        return loss

class LRainrobust(torch.nn.Module):
    """
    rain robust loss, refer to https://visual.ee.ucla.edu/gt_rain.htm/
    """
    def __init__(self, batch_size, n_views, device, temperature=0.07):
        super(LRainrobust, self).__init__()
        self.batch_size = batch_size
        self.n_views = n_views
        self.temperature = temperature
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def forward(self, features):
        logits, labels = self.info_nce_loss(features)
        return self.criterion(logits, labels)

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels


