import copy

from torch import nn
import torch
from model.backbones.vit_pytorch import trunc_normal_
from model.backbones.vit_pytorch import Attention, Block

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


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        # if m.bias:
        nn.init.constant_(m.bias, 0.0)


class multi_stage_vit(nn.Module):
    '''
    1、实现vit
    2、从vit各个支路的输出提取cls
    3、把每个局部特征提取出来进行卷积操作
    '''
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(multi_stage_vit, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.num_classes = num_classes

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        self.base.load_param(model_path)
        print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.act_fun = nn.GELU()
        self.norm = nn.LayerNorm(768)
        self.local_bn = nn.BatchNorm2d(768)

        self.token_self_attention = nn.ModuleList([
            Block(dim=768, num_heads=12, qkv_bias=True)
        for _ in range(6)])


        self.fusion_cnn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels=2*768, out_channels=768, kernel_size=1),
                    nn.BatchNorm2d(768),
                    nn.GELU(),
                    # nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, padding=1, stride=1),
                    # nn.BatchNorm2d(768),
                    # nn.GELU(),
                ) for _ in range(12)
            ]
        )
        self.fusion_cnn.apply(weights_init_kaiming)

        self.bn1 = nn.BatchNorm1d(self.in_planes)
        self.bn1.apply(weights_init_kaiming)
        self.classifier1 = nn.Linear(self.in_planes, self.num_classes, bias=True)
        self.classifier1.apply(weights_init_classifier)

        self.bn2 = nn.BatchNorm1d(768)
        self.bn2.apply(weights_init_kaiming)
        self.classifier2 = nn.Linear(768, self.num_classes, bias=True)
        self.classifier2.apply(weights_init_classifier)

        self.bn3 = nn.BatchNorm1d(768)
        self.bn3.apply(weights_init_kaiming)
        self.classifier3 = nn.Linear(768, self.num_classes, bias=True)
        self.classifier3.apply(weights_init_classifier)

    def forward(self, x, label=None, cam_label=None, view_label=None):

        B = x.shape[0]
        x = self.base.patch_embed(x)

        cls_tokens = self.base.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.base.cam_num > 0 and self.base.view_num > 0:
            x = x + self.base.pos_embed + self.base.sie_xishu * self.base.sie_embed[
                cam_label * self.view_num + view_label]
        elif self.base.cam_num > 0:
            x = x + self.base.pos_embed + self.base.sie_xishu * self.base.sie_embed[cam_label]
        elif self.base.view_num > 0:
            x = x + self.base.pos_embed + self.base.sie_xishu * self.base.sie_embed[view_label]
        else:
            x = x + self.base.pos_embed
        x = self.base.pos_drop(x)


        global_token = []
        for i in range(12):
            x = self.base.blocks[i](x)
            local_token = x[:, 1:].transpose(1,2).reshape(B, -1, 21, 10)
            if i == 0:
                local_feat = self.act_fun(x[:, 1:].transpose(1,2).reshape(B, -1, 21, 10))
            else:
                local_token = torch.cat((local_feat, local_token), 1)
                local_feat = self.act_fun(self.fusion_cnn[i-1](local_token) + local_feat)
            global_token.append(x[:, 0])
        x = self.base.norm(x)

        local_feat = self.local_bn(local_feat)
        local_featvects = self.gap(local_feat).view(B, -1)

        global_feat = torch.stack(global_token, 1)
        for attn in self.token_self_attention:
            global_feat = attn(global_feat)

        global_feat = self.norm(global_feat).mean(1)
        # global_feat = self.norm(global_feat.mean(1))

        if self.training:
            cls = []
            feats = []

            bn_tokens = self.bn1(x[:, 0])
            cls.append(self.classifier1(bn_tokens))
            feats.append(x[:,0])

            bn_glob_feat = self.bn2(global_feat)
            cls.append(self.classifier2(bn_glob_feat))
            feats.append(global_feat)

            bn_local_feat = self.bn3(local_featvects)
            cls.append(self.classifier3(bn_local_feat))
            feats.append(local_featvects)

            return cls, feats
        else:
            return torch.cat((global_feat, local_featvects, x[:,0]), 1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location=torch.device('cpu'))
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))



