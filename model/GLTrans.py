import copy

from torch import nn
import torch
from model.backbones.vit_pytorch import trunc_normal_
from model.backbones.vit_pytorch import Attention, Block
from model.refine_model import minimodule
from model.base_model import first_model

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
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class GLTrans(nn.Module):

    def __init__(self, num_classes, camera_num, view_num, cfg, factory):

        super(GLTrans, self).__init__()
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

        self.base = first_model(camera_num, view_num, cfg, factory)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.act_fun = nn.GELU()
        self.norm = nn.LayerNorm(768)
        self.local_bn = nn.BatchNorm2d(768)

        self.minimodule = minimodule()

        self.bn1 = nn.BatchNorm1d(self.in_planes)
        self.bn1.apply(weights_init_kaiming)
        self.classifier1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier1.apply(weights_init_classifier)

        self.bn_local = nn.BatchNorm1d(3*768)
        self.bn_local.apply(weights_init_kaiming)
        self.classifier_local = nn.Linear(3*768, self.num_classes, bias=False)
        self.classifier_local.apply(weights_init_classifier)

        self.bn_token = nn.ModuleList(nn.BatchNorm1d(self.in_planes) for _ in range(3))
        self.bn_token.apply(weights_init_kaiming)
        self.classifier_clsToken = nn.ModuleList(nn.Linear(768, self.num_classes, bias=False) for _ in range(3))
        self.classifier_clsToken.apply(weights_init_classifier)


    def forward(self, x, label=None, cam_label = None, view_label = None):

        local_tokens, global_tokens = self.base(x, label, cam_label, view_label)
        cls_token = global_tokens[:,-1]
        gf, lf = self.minimodule(local_tokens, global_tokens, cls_token)

        if self.training:
            cls = []
            feats = []

            for i in range(3):
                bn_ClsTokens = self.bn_token[i](global_tokens[:,i])
                cls.append(self.classifier_clsToken[i](bn_ClsTokens))
                feats.append(cls_token)

            bn_gf = self.bn1(gf)
            cls.append(self.classifier1(bn_gf))
            feats.append(gf)

            bn_lf = self.bn_local(lf)
            cls.append(self.classifier_local(bn_lf))
            feats.append(lf)

            return cls, feats

        else:
            # return global_GeMP
            return torch.cat((lf, gf, cls_token), 1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location=torch.device('cpu'))
        for i in self.state_dict().keys():
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))