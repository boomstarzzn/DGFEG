import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale


class GCN(nn.Module):
    def __init__(self, num_state, num_node):
        super(GCN, self).__init__()
        self.num_state = num_state
        self.num_node = num_node
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1)

    def forward(self, seg, aj):
        n, c, h, w = seg.size()
        seg = seg.view(n, self.num_state, -1).contiguous()
        seg_similar = torch.bmm(seg, aj)
        out = self.relu(self.conv2(seg_similar))
        output = out + seg

        return output
def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out


def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))


def make_laplace(img, channels):
    filtered = conv_gauss(img, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up
    return diff


def make_laplace_pyramid(img, level, channels):
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr
class EGP(nn.Module):
    def __init__(self, num_in, plane_mid, mids, normalize=False):
        super(EGP, self).__init__()
        self.num_in = num_in
        self.mids = mids
        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.maxpool_c = nn.AdaptiveAvgPool2d(output_size=(1))
        self.conv_s1 = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_s11 = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_s2 = nn.Conv2d(1, 1, kernel_size=1)
        self.conv_s3 = nn.Conv2d(1, 1, kernel_size=1)
        self.mlp = nn.Linear(num_in, self.num_s)
        self.fc = nn.Conv2d(num_in, self.num_s, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.downsample = nn.AdaptiveAvgPool2d(output_size=(mids, mids))
        self.conv2d = nn.Conv2d(2, 1, 3, 1, 1)
        self.gcn = GCN(num_state=num_in, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1)
        self.blocker = nn.BatchNorm2d(num_in)

    def forward(self, fusion, A,B):
        grayscale_img1 = rgb_to_grayscale(A)
        grayscale_img2 = rgb_to_grayscale(B)
        edge_feature1 = make_laplace_pyramid(grayscale_img1, 5, 1)
        edge_feature2 = make_laplace_pyramid(grayscale_img2, 5, 1)
        edge_feature1 = edge_feature1[1]
        edge_feature2 = edge_feature2[1]
        edge = torch.cat((edge_feature1,edge_feature2),1)
        edge = self.conv2d(edge)
        edge = F.interpolate(edge, size=(64, 64), mode='bilinear', align_corners=True)

        n, c, h, w = fusion.size()
        seg_s = self.conv_s1(fusion)
        theta_T = seg_s.view(n, self.num_s, -1).contiguous()
        theta = seg_s.view(n, -1, self.num_s).contiguous()

        channel_att = torch.relu(self.mlp(self.maxpool_c(fusion).squeeze(3).squeeze(2))).view(n, self.num_s, -1)

        diag_channel_att = torch.bmm(channel_att, channel_att.view(n, -1, self.num_s))
        similarity_c = torch.bmm(theta, diag_channel_att)
        similarity_c = self.softmax(torch.bmm(similarity_c, theta_T))

        seg_c = self.conv_s11(fusion)
        sigma = seg_c.view(n, self.num_s, -1).contiguous()
        sigma_T = seg_c.view(n, -1, self.num_s).contiguous()
        sigma_out = torch.bmm(sigma_T, sigma)

        edge_m = fusion * edge
        maxpool_s, _ = torch.max(fusion, dim=1)
        edge_m_pool, _ = torch.max(edge_m, dim=1)

        fusion_s = self.conv_s2(maxpool_s.unsqueeze(1)).view(n, 1, -1)
        edge_m = self.conv_s3(edge_m_pool.unsqueeze(1)).view(n, -1, 1)

        diag_spatial_att = torch.bmm(edge_m, fusion_s) * sigma_out
        similarity_s = self.softmax(diag_spatial_att)
        similarity = similarity_c + similarity_s

        gcn = self.gcn(fusion, similarity).view(n, self.num_in, self.mids, self.mids)

        finalgcn = gcn + fusion

        return finalgcn











