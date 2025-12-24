import copy
from torch import nn
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from model.backbones.vit_pytorch import Attention, Block, trunc_normal_



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('LayerNorm') != -1:
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        # if m.bias:
        nn.init.constant_(m.bias, 0.0)


class minimodule(nn.Module):

    def __init__(self):
        super(minimodule, self).__init__()

        self.in_planes = 768

        # local
        self.global_interaction = nn.Sequential(
                    nn.Linear(in_features=3*768, out_features=768),
                    # nn.GELU()
                )

        self.local_interaction0 = nn.ModuleList(
            Block(
                dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True,
                drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm)
            for _ in range(3)
            )

        self.local_interaction1 = nn.ModuleList(
            Block(
                dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True,
                drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm)
            for _ in range(3)
            )

        self.local_interaction_norm = nn.LayerNorm(768)

        self.downchannel = nn.Sequential(
            nn.Conv2d(in_channels=3*768, out_channels=768, kernel_size=1, stride=1),
            nn.GELU(),
            nn.BatchNorm2d(768),
            nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, padding=1, stride=1),
        )

        self.G2L = G2LInteraction(dim=768, num_heads=12)
        self.u_cls = nn.Parameter(torch.zeros(1, 1, 768))
        self.m_cls = nn.Parameter(torch.zeros(1, 1, 768))
        self.l_cls = nn.Parameter(torch.zeros(1, 1, 768))

        self.apply(weights_init_kaiming)
        # trunc_normal_(self.u_cls, std=.02)
        # trunc_normal_(self.m_cls, std=.02)
        # trunc_normal_(self.l_cls, std=.02)



    def forward(self, local_feats, global_feats, cls_token=None):

        b = local_feats.size(0)

        # global
        g_f = global_feats.view(global_feats.size(0), -1)
        g_f = self.global_interaction(g_f)

        # local
        l_f = self.downchannel(local_feats).reshape(b, -1, 210).transpose(1, 2)
        l_f = self.G2L(g_f, l_f).transpose(1,2).reshape(b, -1, 21, 10)

        u_cls = self.u_cls.expand(b, -1, -1)
        m_cls = self.m_cls.expand(b, -1, -1)
        l_cls = self.l_cls.expand(b, -1, -1)

        upper_lf = torch.cat((u_cls, global_feats, l_f[:, :, :7].view(b, 768, 70).transpose(1,2)), 1)
        middle_lf = torch.cat((m_cls, global_feats, l_f[:, :, 7:14].view(b, 768, 70).transpose(1,2)), 1)
        lower_lf = torch.cat((l_cls, global_feats, l_f[:, :, 14:].view(b, 768, 70).transpose(1,2)), 1)

        upper_lf = self.local_interaction0[0](upper_lf)
        middle_lf = self.local_interaction0[1](middle_lf)
        lower_lf = self.local_interaction0[2](lower_lf)

        upper_lf = self.local_interaction1[0](upper_lf)
        middle_lf = self.local_interaction1[1](middle_lf)
        lower_lf = self.local_interaction1[2](lower_lf)

        upper_lf = self.local_interaction_norm(upper_lf)
        middle_lf = self.local_interaction_norm(middle_lf)
        lower_lf = self.local_interaction_norm(lower_lf)

        upper_cls = upper_lf[:, 0]
        middle_cls = middle_lf[:, 0]
        lower_cls = lower_lf[:, 0]
        lf = torch.cat((upper_cls, middle_cls, lower_cls), 1)

        return g_f, lf

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location=torch.device('cpu'))
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

class G2LInteraction(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.GLinear = nn.Linear(in_features=dim, out_features=dim)
        self.LLinear = nn.Linear(in_features=dim, out_features=dim)

    def forward(self, gf, lf):

        B, N, C = lf.shape
        gf = gf.unsqueeze(1)
        mapping_gf = self.GLinear(gf).reshape(B, 1, self.num_heads, C // self.num_heads).transpose(1,2)
        mapping_lf = self.LLinear(lf).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1,2)
        lf_ = lf.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = (mapping_lf @ mapping_gf.transpose(-2, -1))
        attn = torch.sigmoid(attn)
        lf = (lf_ * attn).transpose(1, 2).reshape(B, N, C) + lf

        return lf