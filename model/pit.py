import copy

from torch import nn
import torch
from model.backbones.vit_pytorch import trunc_normal_
from .backbones.vit_pytorch import Block, Attention


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


class part_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(part_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

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
                                                        drop_rate=cfg.MODEL.DROP_OUT, local_feature=True,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        print('==Define the global transformer layer==')
        self.global_trans = nn.Sequential(
            copy.deepcopy(self.base.blocks[-1]),
            copy.deepcopy(self.base.norm)
        )
        self.global_part_token = nn.Parameter(torch.zeros(1, 1, self.in_planes))
        trunc_normal_(self.global_part_token, std=.02)

        print('==Define the horizontal tranformer layer==')
        self.horizontal_transformer = nn.Sequential(
            copy.deepcopy(self.base.blocks[-1]),
            copy.deepcopy(self.base.norm)
        )
        self.horizontal_part_token = nn.Parameter(torch.zeros(1, 1, self.in_planes))
        trunc_normal_(self.horizontal_part_token, std=.02)

        print('==Define the vertical tranformer layer==')
        self.vertical_transformer = nn.Sequential(
            copy.deepcopy(self.base.blocks[-1]),
            copy.deepcopy(self.base.norm)
        )
        self.vertical_part_token = nn.Parameter(torch.zeros(1, 1, self.in_planes))
        trunc_normal_(self.vertical_part_token, std=.02)

        print('==Define the patch tranformer layer==')
        self.patch_transformer = nn.Sequential(
            copy.deepcopy(self.base.blocks[-1]),
            copy.deepcopy(self.base.norm)
        )
        self.patch_part_token = nn.Parameter(torch.zeros(1, 1, self.in_planes))
        trunc_normal_(self.patch_part_token, std=.02)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cls_token_trans = token_MSA()

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.bn = nn.ModuleList([nn.BatchNorm1d(self.in_planes) for _ in range(1)])
        self.classifier = nn.ModuleList([nn.Linear(self.in_planes, num_classes) for _ in range(1)])
        # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        for j in range(len(self.classifier)):
            self.classifier[j].apply(weights_init_classifier)
        self.classifier.apply(weights_init_classifier)

        # self.bottleneck = nn.BatchNorm1d(self.in_planes)
        for i in range(len(self.bn)):
            self.bn[i].bias.requires_grad_(False)
            self.bn[i].apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)
        # check一下，这里需不需要用norm。可能差别还是有点大的！
        # global_feat = self.base.norm(global_feat)
        b, n, c = global_feat.size()
        token = global_feat[:, 0]

        # global stream
        # global_part_token = self.global_part_token.expand(b, -1, -1)
        # global_part_feat = torch.cat((global_part_token, global_feat), 1)
        global_part_feat = self.global_trans(global_feat)
        # global_part_feat = self.base.norm(global_part_feat)
        global_part_cls = global_part_feat[:, 0, :].unsqueeze(1)

        # horizontal stream
        horizontal_part_feat = global_feat[:, 1:].reshape(b, 3, 70, c)
        # horizontal_part_feat = torch.cat((token.view(b, 1, 1, -1).repeat(1, 3, 1, 1), horizontal_part_feat), 2)
        horizontal_part_token = token.reshape(b, 1, 1, -1).repeat(1, 3, 1, 1)
        horizontal_part_feat = torch.cat((horizontal_part_token, horizontal_part_feat), 2).reshape(b * 3, 71, c)
        horizontal_part_feat = self.horizontal_transformer(horizontal_part_feat).reshape(b, 3, 71, c)
        # horizontal_part_feat = self.base.norm(horizontal_part_feat)
        horizontal_part_cls = horizontal_part_feat[:, :, 0, :]

        # vertical stream
        vertical_part_feat = global_feat[:, 1:].reshape(b, self.base.patch_embed.num_y, self.base.patch_embed.num_x, -1)
        vertical_part_feat = vertical_part_feat.transpose(1, 2).reshape(b, 2, 105, c)
        # vertical_part_feat = torch.cat((token.view(b, 1, 1, -1).repeat(1, 2, 1, 1), vertical_part_feat), 2)
        vertical_part_token = token.reshape(b, 1, 1, -1).repeat(1, 2, 1, 1)
        vertical_part_feat = torch.cat(
            (vertical_part_token, vertical_part_feat), 2
        ).reshape(b * 2, 106, c)
        vertical_part_feat = self.vertical_transformer(vertical_part_feat).reshape(b, 2, 106, c)
        # vertical_part_feat = self.base.norm(vertical_part_feat)
        vertical_part_cls = vertical_part_feat[:, :, 0, :]

        # patch stream
        patch_concat_matrix = []
        reshape_feat = global_feat[:, 1:].reshape(b, self.base.patch_embed.num_y, self.base.patch_embed.num_x, c)
        for i in range(7):
            patch_concat = reshape_feat[:, i::7, :, :].reshape(b, int(self.base.patch_embed.num_y / 7) * int(
                self.base.patch_embed.num_x / 5), -1, c)
            patch_concat_matrix.append(patch_concat)
        patch_part_feat = torch.cat(patch_concat_matrix, 2)  # b, 6, 35, 768

        # patch_part_feat = torch.cat((token.view(b, 1, 1, -1).repeat(1, 6, 1, 1), patch_part_feat), 2)
        patch_part_token = token.reshape(b, 1, 1, -1).repeat(1, 6, 1, 1)
        patch_part_feat = torch.cat(
            (patch_part_token, patch_part_feat), 2
        ).reshape(b * 6, 36, 768)
        patch_part_feat = self.patch_transformer(patch_part_feat).reshape(-1, 6, 36, 768)
        # patch_part_feat = self.base.norm(patch_part_feat)
        patch_part_cls = patch_part_feat[:, :, 0, :]

        cls = torch.cat((global_part_cls, vertical_part_cls, horizontal_part_cls, patch_part_cls), 1)
        cls = self.cls_token_trans(cls)

        if self.training:
            # cls_score = []
            # f_list = []
            # for num in range(1):
            #     bn_f = self.bn[num](cls)
            #     cls_score.append(self.classifier[num](bn_f))
            #     f_list.append(cls)

            return self.classifier[0](self.bn[0](cls)), cls  # global feature for triplet loss
        else:
            return cls.reshape(b, -1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location=torch.device('cpu'))
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class token_MSA(nn.Module):

    def __init__(self, dim=768, num_heads=12, qkv_bias=True, mlp_ratio=4.0, norm_layer=nn.LayerNorm, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=dim, out_features=mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(in_features=mlp_hidden_dim, out_features=dim),
        )

    def forward(self, x):
        B, N, C = x.shape
        q_x = x[:, 0]
        kv_x = x[:, 1:]
        kv = self.kv(kv_x).reshape(B, N - 1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q(q_x).reshape(B, 1, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = self.proj((attn @ v).transpose(1, 2).reshape(B, C))
        x = self.mlp(x)

        return x