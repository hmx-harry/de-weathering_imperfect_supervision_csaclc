from models.batch_gatnet import *
import warnings
from models.common import vgg_19, Conv, ResBlock, UpSample, UpDysample
from utils.dysample import *


warnings.filterwarnings("ignore")

def pad_graph(feature_map, multiple):
    original_height, original_width = feature_map.size(2), feature_map.size(3)

    target_height = ((original_height + multiple - 1) // multiple) * multiple
    target_width = ((original_width + multiple - 1) // multiple) * multiple

    padding_height = target_height - original_height
    padding_width = target_width - original_width

    padded_feature_map = F.pad(feature_map, (0, padding_width, 0, padding_height))

    return padded_feature_map

class CSAModule(nn.Module):
    def __init__(self, w_siz, w_step, expend_scale, pad_siz, topk=3):
        super(CSAModule, self).__init__()
        self.w_siz = w_siz
        self.w_step = w_step
        self.expend_scale = expend_scale
        self.topk = topk
        self.pad_siz = pad_siz

        self.GAT = GAT(n_feat=2304, n_hid=128, n_class=2304, dropout=0.1, alpha=1, n_heads=2)
    def find_correlation(self, im, adj_frames):
        b, c, w, h = im.shape

        parents = torch.nn.Unfold(kernel_size=self.w_siz[0],
                                      stride=self.w_step, padding=1)(im)
        parent_coll = (parents.contiguous()
                   .view(parents.shape[0], im.shape[1], self.w_siz[0], self.w_siz[1], -1)
                   .permute(4, 0, 1, 2, 3)).unsqueeze(1)

        frame_coors = []
        for idx in range(adj_frames.shape[0]):
            region = torch.nn.Unfold(kernel_size=self.w_siz[0] + self.expend_scale*2,
                                     stride=self.w_step, padding=self.expend_scale+1)(adj_frames[idx])
            region = (region.contiguous()
                      .view(region.shape[0], im.shape[1],
                            self.w_siz[0] + self.expend_scale*2, self.w_siz[1] + self.expend_scale*2, -1)
                      .permute(4, 0, 1, 2, 3))

            regions_coors = []
            for prt_idx in range(parent_coll.shape[0]):
                patches = torch.nn.Unfold(kernel_size=self.w_siz[0],
                                          stride=self.w_step, padding=0)(region[prt_idx])
                patches = (patches.contiguous().view(patches.shape[0], im.shape[1],
                                                     self.w_siz[0], self.w_siz[1], -1)
                           .permute(4, 0, 1, 2, 3))
                corr = self.filter_topk_corr(parent_coll[prt_idx], patches, topk=self.topk)
                regions_coors.append(corr)
            regions_coors = torch.stack(regions_coors, dim=0)
            frame_coors.append(regions_coors)
        childs_coll = torch.cat(frame_coors, dim=1)
        return parent_coll, childs_coll
    def filter_topk_corr(self, parent, patches, topk):
        similarity = []
        flat_parent = parent.contiguous().view(parent.shape[0], -1)
        flat_patches = patches.contiguous().view(patches.shape[0], -1)

        for i in range(flat_patches.shape[0]):
            val = torch.mean(torch.cosine_similarity(flat_parent, flat_patches[i], dim=0))
            similarity.append(val)
        _, idx = torch.topk(torch.stack(similarity), topk)
        return patches[idx]

    def gen_edge_matrix(self, num_node):
        adj = torch.zeros((num_node, num_node), dtype=torch.float32)
        adj[1:, 0] = 1
        return adj

    def forward(self, multi_fea):
        x_fea, x_prev_fea, x_next_fea = torch.split(multi_fea, multi_fea.shape[0]//3, dim=0)
        x_adj_fea = torch.stack([x_prev_fea, x_next_fea], dim=0)

        x_fea_pad = pad_graph(x_fea, self.w_siz[0])
        x_adj_fea_pad = torch.stack([pad_graph(element, self.w_siz[0]) for element in x_adj_fea])

        node_matrix, edge_matrix = self.create_graphs(x_fea_pad, x_adj_fea_pad)

        batch_gatout = []
        for idx in range(node_matrix.shape[0]):
            gatout = self.GAT(x=node_matrix[idx], adj=edge_matrix)
            batch_gatout.append(gatout)
        batch_gatout = torch.stack(batch_gatout, dim=0).permute(0,2,1)

        batch_gatout = torch.nn.Fold(output_size=(x_fea_pad.shape[2],x_fea_pad.shape[3]),
                                     kernel_size=(3,3), stride=3)(batch_gatout)

        batch_gatout = batch_gatout[:, :, :x_fea.shape[2], :x_fea.shape[3]]
        return batch_gatout

    def create_graphs(self, im, im_adj):
        parent_coll, childs_coll = self.find_correlation(im, im_adj)

        node_matrix = torch.cat([parent_coll, childs_coll], dim=1)
        num_node = node_matrix.shape[1]
        node_matrix = node_matrix.permute(2, 0, 1, 3, 4, 5)
        node_matrix = node_matrix.view(node_matrix.shape[0], node_matrix.shape[1],
                                       node_matrix.shape[2], -1).to(im.device)

        edge_matrix = self.gen_edge_matrix(num_node=num_node).to(im.device)

        return node_matrix, edge_matrix

class CSACLC(nn.Module):
    def __init__(self):
        super(CSACLC, self).__init__()
        self.n_blocks = 4
        self.fea_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(start_dim=1, end_dim=-1)
        )
        self.GAM = CSAModule(w_siz=[3, 3], w_step=3, expend_scale=3, topk=3, pad_siz=1)

        self.conv_in1 = Conv(in_ch=3, out_ch=64, k_size=7, stride=1)
        self.conv_in2 = Conv(in_ch=64, out_ch=64, k_size=3, stride=1)
        self.conv_down1 = Conv(in_ch=64, out_ch=128, k_size=3, stride=2)
        self.conv_down2 = Conv(in_ch=128, out_ch=256, k_size=3, stride=2)
        self.conv_down3 = Conv(in_ch=256, out_ch=512, k_size=3, stride=2)

        self.res_blocks = nn.Sequential()
        for idx in range(self.n_blocks):
            self.res_blocks.add_module(f'res_block_{idx}',
                                       ResBlock(in_ch=512, out_ch=512, k_size=3, stride=1))

        self.conv_down1d = Conv(in_ch=512, out_ch=256, k_size=1, stride=1)
        self.conv_up1d = Conv(in_ch=256, out_ch=512, k_size=1, stride=1)
        self.conv_decode1 = Conv(in_ch=512, out_ch=512, k_size=3, stride=1)

        self.conv_decode2 = Conv(in_ch=512*4, out_ch=512, k_size=1, stride=1)
        self.conv_up1 = UpSample(scale=2, in_ch=512, out_ch=256)
        self.conv_decode3 = Conv(in_ch=256+256, out_ch=256, k_size=3, stride=1)
        self.conv_up2 = UpSample(scale=2, in_ch=256, out_ch=128)
        self.conv_decode4 = Conv(in_ch=128+128, out_ch=128, k_size=3, stride=1)
        self.conv_up3 = UpSample(scale=2, in_ch=128, out_ch=64)
        self.conv_decode5 = Conv(in_ch=64+64, out_ch=64, k_size=3, stride=1)

        self.graph_skip = UpDysample(scale=8, in_ch=512, out_ch=64)

        self.conv_out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3,
                                  stride=1, padding=1, bias=True)


    def forward(self, x, x_coll):
        x_aid1, x_aid2, clean = x_coll
        x_in = torch.cat([x, x_aid1, x_aid2, clean], dim=0)
        x1 = self.conv_in1(x_in)
        x2 = self.conv_in2(x1)
        x3 = self.conv_down1(x2)
        x4 = self.conv_down2(x3)
        x5 = self.conv_down3(x4)
        mix_fea = self.res_blocks(x5)

        x_fea, x_aid1_fea, x_aid2_fea, clean_fea = torch.split(mix_fea, x.shape[0], dim=0)
        graph_in = torch.cat([x_fea, x_aid1_fea, x_aid2_fea], dim=0)

        graph_in = self.conv_down1d(graph_in)

        graph_out = self.GAM(graph_in)
        graph_out = self.conv_up1d(graph_out)
        x_fea_g = torch.mul(x_fea, graph_out)

        x_fea_g = self.conv_decode1(x_fea_g)
        x6 = self.conv_decode2(torch.cat([x_fea_g, x_aid1_fea, x_aid2_fea, clean_fea], dim=1))
        x7 = self.conv_up1(x6)
        x8 = self.conv_decode3(torch.cat([x7, x4[:x.shape[0]]], dim=1))
        x9 = self.conv_up2(x8)
        x10 = self.conv_decode4(torch.cat([x9, x3[:x.shape[0]]], dim=1))
        x11 = self.conv_up3(x10)
        x12 = self.conv_decode5(torch.cat([x11, x2[:x.shape[0]]], dim=1))

        graph_out_up = self.graph_skip(graph_out)

        out = F.tanh(self.conv_out(torch.mul(x12, graph_out_up)))

        x_clean_proj = self.fea_proj(torch.cat([x_fea, clean_fea], dim=0))
        return out, x_clean_proj, x_fea_g

class Dewea_train(nn.Module):
    def __init__(self):
        super(Dewea_train, self).__init__()
        self.n_blocks = 0
        self.fea_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(start_dim=1, end_dim=-1)
        )

        self.conv_in1 = Conv(in_ch=3, out_ch=64, k_size=7, stride=1)
        self.conv_in2 = Conv(in_ch=64, out_ch=64, k_size=3, stride=1)
        self.conv_down1 = Conv(in_ch=64, out_ch=128, k_size=3, stride=2)
        self.conv_down2 = Conv(in_ch=128, out_ch=256, k_size=3, stride=2)
        self.conv_down3 = Conv(in_ch=256, out_ch=512, k_size=3, stride=2)

        self.res_blocks = nn.Sequential()
        for idx in range(self.n_blocks):
            self.res_blocks.add_module(f'res_block_{idx}',
                                       ResBlock(in_ch=512, out_ch=512, k_size=3, stride=1))

        self.conv_up1 = UpSample(scale=2, in_ch=512, out_ch=256)
        self.conv_decode1 = Conv(in_ch=256+256, out_ch=256, k_size=3, stride=1)
        self.conv_up2 = UpSample(scale=2, in_ch=256, out_ch=128)
        self.conv_decode2 = Conv(in_ch=128+128, out_ch=128, k_size=3, stride=1)
        self.conv_up3 = UpSample(scale=2, in_ch=128, out_ch=64)
        self.conv_decode3 = Conv(in_ch=64+64, out_ch=64, k_size=3, stride=1)

        self.conv_out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3,
                                  stride=1, padding=1, bias=True)

    def forward(self, x, clean):
        x_in = torch.cat([x, clean], dim=0)
        x1 = self.conv_in1(x_in)
        x2 = self.conv_in2(x1)
        x3 = self.conv_down1(x2)
        x4 = self.conv_down2(x3)
        x5 = self.conv_down3(x4)
        mix_fea = self.res_blocks(x5)
        x_fea, _ = torch.split(mix_fea, x.shape[0], dim=0)

        x6 = self.conv_up1(x_fea)
        x7 = self.conv_decode1(torch.cat([x6, x4[:x.shape[0]]], dim=1))
        x8 = self.conv_up2(x7)
        x9 = self.conv_decode2(torch.cat([x8, x3[:x.shape[0]]], dim=1))
        x10 = self.conv_up3(x9)
        x11 = self.conv_decode3(torch.cat([x10, x2[:x.shape[0]]], dim=1))
        out = F.tanh(self.conv_out(x11))

        x_clean_proj = self.fea_proj(mix_fea)
        return out, x_clean_proj, x_fea


class Dewea_test(nn.Module):
    def __init__(self):
        super(Dewea_test, self).__init__()
        self.n_blocks = 0
        self.fea_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(start_dim=1, end_dim=-1)
        )

        self.conv_in1 = Conv(in_ch=3, out_ch=64, k_size=7, stride=1)
        self.conv_in2 = Conv(in_ch=64, out_ch=64, k_size=3, stride=1)
        self.conv_down1 = Conv(in_ch=64, out_ch=128, k_size=3, stride=2)
        self.conv_down2 = Conv(in_ch=128, out_ch=256, k_size=3, stride=2)
        self.conv_down3 = Conv(in_ch=256, out_ch=512, k_size=3, stride=2)

        self.res_blocks = nn.Sequential()
        for idx in range(self.n_blocks):
            self.res_blocks.add_module(f'res_block_{idx}',
                                       ResBlock(in_ch=512, out_ch=512, k_size=3, stride=1))

        self.conv_up1 = UpSample(scale=2, in_ch=512, out_ch=256)
        self.conv_decode1 = Conv(in_ch=256+256, out_ch=256, k_size=3, stride=1)
        self.conv_up2 = UpSample(scale=2, in_ch=256, out_ch=128)
        self.conv_decode2 = Conv(in_ch=128+128, out_ch=128, k_size=3, stride=1)
        self.conv_up3 = UpSample(scale=2, in_ch=128, out_ch=64)
        self.conv_decode3 = Conv(in_ch=64+64, out_ch=64, k_size=3, stride=1)

        self.conv_out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3,
                                  stride=1, padding=1, bias=True)

    def forward(self, x):
        x1 = self.conv_in1(x)
        x2 = self.conv_in2(x1)
        x3 = self.conv_down1(x2)
        x4 = self.conv_down2(x3)
        x5 = self.conv_down3(x4)
        mix_fea = self.res_blocks(x5)

        x6 = self.conv_up1(mix_fea)
        x7 = self.conv_decode1(torch.cat([x6, x4[:x.shape[0]]], dim=1))
        x8 = self.conv_up2(x7)
        x9 = self.conv_decode2(torch.cat([x8, x3[:x.shape[0]]], dim=1))
        x10 = self.conv_up3(x9)
        x11 = self.conv_decode3(torch.cat([x10, x2[:x.shape[0]]], dim=1))
        out = F.tanh(self.conv_out(x11))

        return out