import torch
from torch import nn
import copy
from torch.nn import functional as F
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID


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

class first_model(nn.Module):

    def __init__(self, camera_num, view_num, cfg, factory):
        super(first_model, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        # print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        self.base.load_param(model_path)
        print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.PAM = nn.ModuleList(
            PatchAttentionModule(inplanes=self.in_planes, ratio=4) for _ in range(3)
        )
        self.PAM.apply(weights_init_kaiming)

        self.norm = nn.ModuleList(
            copy.deepcopy(self.base.norm) for _ in range(3)
        )

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location=torch.device('cpu'))
        for i in param_dict:
            if 'classifier' in i:
                continue
            if 'bottleneck' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def forward(self, x, label=None, cam_label= None, view_label=None):
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

        global_tokens = []
        local_tokens = []

        for i in range(12):
            x = self.base.blocks[i](x)
            if i >= 9:
                norm_token = self.norm[i - 9](x)
                # local_token = norm_token[:, 1:].transpose(1, 2)
                local_token = self.PAM[i - 9](x[:,1:].transpose(1,2)).reshape(B, -1, 21, 10)
                local_tokens.append(local_token)
                global_tokens.append((norm_token[:, 0]))

        # x = self.base.norm(x)
        local_tokens = torch.cat(local_tokens, 1)
        global_tokens = torch.stack(global_tokens, 1)

        return local_tokens, global_tokens,


class PatchAttentionModule(nn.Module):

    def __init__(self, ratio, inplanes):

        super(PatchAttentionModule, self).__init__()
        mid_channel = int(inplanes / ratio)
        # self.linear1 = nn.Linear(in_features=inplanes, out_features=mid_channel)
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.act_layer1 = nn.GELU()
        self.bn1 = nn.BatchNorm2d(mid_channel)
        # self.linear2 = nn.Linear(in_features=mid_channel, out_features=inplanes)
        self.conv2 = nn.Conv2d(in_channels=mid_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.act_layer2 = nn.Sigmoid()


    def forward(self, x):
        x = x.reshape(-1, 768, 21, 10)
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_layer1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act_layer2(x)
        out = residual * x + residual
        out = out.reshape(-1, 768, 21*10)
        return out
