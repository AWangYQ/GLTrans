import copy

from PIL import Image, ImageFile
import torchvision.transforms as T
from timm.data.random_erasing import RandomErasing
import os.path as osp
import torch
from torch import nn
from model.backbones.vit_pytorch import trunc_normal_
from model.backbones.vit_pytorch import Attention, Block, vit_base_patch16_224_TransReID
from loss.triplet_loss import TripletLoss, hard_example_mining

def load_test_samples(img_path_list):
    """
        根据图片的路径得到对应的标签和摄像头标签
    """

    img_list = []
    pids = []
    camids = []

    for img_path in img_path_list:

        pid = int(img_path.split('/')[6])
        camid = int(img_path.split('/')[7].split('_')[1])
        pids.append(pid)
        camids.append(camid)
        img_list.append(img_path)

    return pids, camids


def transform():
    """
    定义数据预处理的方法
    """
    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_transforms = T.Compose([
        T.Resize([256, 128], interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=0.5, mode='pixel', max_count=1, device='cpu'),
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])

    return val_transforms, train_transforms

def read_image(img_path):

    """
    Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process.
    读取图片矩阵
    """

    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def transforme_images(img_path_list, transforms):

    """
    处理图片后，将其转换成torh.Tensor格式
    """
    imgs = []
    for img_path in img_path_list:
        img = read_image(img_path)
        img = transforms(img)
        imgs.append(img)

    imgs = torch.stack(imgs, 0)

    return imgs

def define_model():
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

        def __init__(self):
            super(multi_stage_vit, self).__init__()
            model_path = '/root/autodl-tmp/pre-trained model/jx_vit_base_p16_224-80ecf9dd.pth'
            self.in_planes = 768
            self.num_classes = 1041

            print('using Transformer_type: {} as a backbone'.format('transreid'))

            self.base = vit_base_patch16_224_TransReID(img_size=(256, 128), sie_xishu=3.0,
                                                       camera=15, view=0,
                                                       stride_size=[12, 12],
                                                       drop_path_rate=0.1,
                                                       drop_rate=0.0,
                                                       attn_drop_rate=0.0)
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
                        nn.Conv2d(in_channels=2 * 768, out_channels=768, kernel_size=1),
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
                local_token = x[:, 1:].transpose(1, 2).reshape(B, -1, 21, 10)
                if i == 0:
                    local_feat = self.act_fun(x[:, 1:].transpose(1, 2).reshape(B, -1, 21, 10))
                else:
                    local_token = torch.cat((local_feat, local_token), 1)
                    local_feat = self.act_fun(self.fusion_cnn[i - 1](local_token) + local_feat)
                global_token.append(x[:, 0])

            x = self.base.norm(x)

            local_feat = self.local_bn(local_feat)
            local_featvects = self.gap(local_feat).view(B, -1)

            global_feat = torch.stack(global_token, 1)
            for attn in self.token_self_attention:
                global_feat = attn(global_feat)

            global_feat = self.norm(global_feat).mean(1)

            cls = []
            feats = []

            bn_tokens = self.bn1(x[:, 0])
            cls.append(self.classifier1(bn_tokens))
            feats.append(x[:, 0])

            bn_glob_feat = self.bn2(global_feat)
            cls.append(self.classifier2(bn_glob_feat))
            feats.append(global_feat)

            bn_local_feat = self.bn3(local_featvects)
            cls.append(self.classifier3(bn_local_feat))
            feats.append(local_featvects)

            return cls, feats

        def load_param(self, trained_path):
            param_dict = torch.load(trained_path, map_location=torch.device('cpu'))
            for k,v in self.state_dict().items():
                if k not in param_dict:
                    print(k)
                    break
            print('=========')

            for i in param_dict:
                self.state_dict()[i].copy_(param_dict[i])
            print('Loading pretrained model from {}'.format(trained_path))

    model = multi_stage_vit()
    model.load_param('/root/autodl-tmp/outputs/new_idea/2022-11-30_23-09-50/new_idea_120.pth')

    return model

def euclidean_dist_(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

if __name__ == '__main__':
    img_path_list = ['/root/autodl-tmp/datasets/MSMT17/train/0013/0013_003_01_0303morning_0164_1_ex.jpg', 
                      '/root/autodl-tmp/datasets/MSMT17/train/0952/0952_003_01_0114afternoon_0398_0_ex.jpg',
                     '/root/autodl-tmp/datasets/MSMT17/train/0952/0952_005_01_0114afternoon_0399_0_ex.jpg',
                     '/root/autodl-tmp/datasets/MSMT17/train/0008/0008_009_05_0303morning_0078_1.jpg',
                     '/root/autodl-tmp/datasets/MSMT17/train/0008/0008_004_01_0303morning_0074_1_ex.jpg',
                     '/root/autodl-tmp/datasets/MSMT17/train/0013/0013_005_14_0303morning_0391_1_ex.jpg']
    
    pids, camids = load_test_samples(img_path_list)
    pids = torch.Tensor(pids)
    val_transforms, train_transforms = transform()
    imgs = transforme_images(img_path_list, train_transforms)
    model = define_model()
    
    model.eval()
    with torch.no_grad():
        cls, feats = model(imgs, pids, camids)

    qf = feats[0][0,:].unsqueeze(0)
    gf = feats[0][1:,:]
    m = qf.shape[0]
    n = gf.shape[0]
    #
    # print("The test feature is normalized")
    # qf = torch.nn.functional.normalize(qf, dim=1, p=2)
    # gf = torch.nn.functional.normalize(gf, dim=1, p=2)
    # 计算两个特征向量的欧氏距离：
    euclidean_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    euclidean_dist.addmm_(1, -2, qf, gf.t())
    # euclidean_dist.addmm_(qf, gf.t(), 1, -2)
    # print("euclidean_distance is :", euclidean_dist)

    # dist = euclidean_dist_(feats[0], feats[0])
    # print("euclidean_distance matrix is :" ,dist)
    triplet = TripletLoss()
    TRI_LOSS = triplet(feats[0], pids)
    print(TRI_LOSS[0])
    print(TRI_LOSS[1])
    print(TRI_LOSS[2])
    print(TRI_LOSS[3])




    