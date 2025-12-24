import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import os.path as osp
import cv2
import numpy as np

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = 0.
                for i in range(len(score)):
                    acc = acc + (score[i].max(1)[1] == target).float().mean()
                acc = acc / len(score)
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()
            img_path_list = []
            for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.cuda()
                    camids = camids.cuda()
                    target_view = target_view.cuda()
                    feat = model(img, cam_label=camids, view_label=target_view)
                    evaluator.update((feat, pid, camid, imgpath))
                    img_path_list.extend(imgpath)

            cmc, mAP, _, _, _, _, _, query_img_paths, gallery_img_paths = evaluator.compute()
            logger.info("Validation Results ")
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

            if cfg.TEST.SHOW_BADSAMPLE :

                dst = osp.join(cfg.OUTPUT_DIR, 'bad retrieve samples')
                if not os.path.exists(dst):
                    os.mkdir(dst)

                GRID_SPACING = 10
                QUERY_EXTRA_SPACING = 90
                BW = 5  # border width
                GREEN = (0, 255, 0)
                RED = (0, 0, 255)
                topk = 11
                height, width = cfg.INPUT.SIZE_TRAIN

                for q_idx in range(len(query_img_paths)):
                    q_imname = osp.basename(osp.splitext(query_img_paths[q_idx])[0])
                    qpid = q_imname.split('_')[0]
                    qimg = cv2.imread(query_img_paths[q_idx])
                    qimg = cv2.resize(qimg, (width, height))
                    qimg = cv2.copyMakeBorder(
                        qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )
                    # resize twice to ensure that the border width is consistent across images
                    qimg = cv2.resize(qimg, (width, height))
                    num_cols = topk + 1
                    grid_img = 255 * np.ones(
                        (
                            height,
                            num_cols * width + topk * GRID_SPACING + QUERY_EXTRA_SPACING, 3
                        ),
                        dtype=np.uint8
                    )
                    grid_img[:, :width, :] = qimg

                    rank_idx = 1
                    for g_idx in range(len(gallery_img_paths[q_idx])):
                        g_imname = osp.basename(osp.splitext(gallery_img_paths[q_idx][g_idx])[0])
                        gpid = g_imname.split('_')[0]
                        matched = gpid == qpid
                        border_color = GREEN if matched else RED
                        gimg = cv2.imread(gallery_img_paths[q_idx][g_idx])
                        gimg = cv2.resize(gimg, (width, height))
                        gimg = cv2.copyMakeBorder(
                            gimg,
                            BW,
                            BW,
                            BW,
                            BW,
                            cv2.BORDER_CONSTANT,
                            value=border_color
                            # value = (0, 0, 0)
                        )
                        gimg = cv2.resize(gimg, (width, height))
                        start = rank_idx * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                        end = (
                                      rank_idx + 1
                              ) * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                        grid_img[:, start:end, :] = gimg
                        rank_idx += 1

                    cv2.imwrite(osp.join(dst, q_imname + '.jpg'), grid_img)


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.cuda()
            camids = camids.cuda()
            target_view = target_view.cuda()
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid, imgpath))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _, query_img_paths, gallery_img_paths = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    if cfg.TEST.SHOW_BADSAMPLE:

        dst = osp.join(cfg.OUTPUT_DIR, 'bad retrieve samples')
        if not os.path.exists(dst):
            os.mkdir(dst)

        GRID_SPACING = 10
        QUERY_EXTRA_SPACING = 90
        BW = 5  # border width
        GREEN = (0, 255, 0)
        RED = (0, 0, 255)
        topk = 11
        height, width = cfg.INPUT.SIZE_TRAIN

        for q_idx in range(len(query_img_paths)):
            q_imname = osp.basename(osp.splitext(query_img_paths[q_idx])[0])
            qpid = q_imname.split('_')[0]
            qimg = cv2.imread(query_img_paths[q_idx])
            qimg = cv2.resize(qimg, (width, height))
            qimg = cv2.copyMakeBorder(
                qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            # resize twice to ensure that the border width is consistent across images
            qimg = cv2.resize(qimg, (width, height))
            num_cols = topk + 1
            grid_img = 255 * np.ones(
                (
                    height,
                    num_cols * width + topk * GRID_SPACING + QUERY_EXTRA_SPACING, 3
                ),
                dtype=np.uint8
            )
            grid_img[:, :width, :] = qimg

            rank_idx = 1
            for g_idx in range(len(gallery_img_paths[q_idx])):
                g_imname = osp.basename(osp.splitext(gallery_img_paths[q_idx][g_idx])[0])
                gpid = g_imname.split('_')[0]
                matched = gpid == qpid
                border_color = GREEN if matched else RED
                gimg = cv2.imread(gallery_img_paths[q_idx][g_idx])
                gimg = cv2.resize(gimg, (width, height))
                gimg = cv2.copyMakeBorder(
                    gimg,
                    BW,
                    BW,
                    BW,
                    BW,
                    cv2.BORDER_CONSTANT,
                    value=border_color
                    # value = (0, 0, 0)
                )
                gimg = cv2.resize(gimg, (width, height))
                start = rank_idx * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                end = (
                              rank_idx + 1
                      ) * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                grid_img[:, start:end, :] = gimg
                rank_idx += 1

            cv2.imwrite(osp.join(dst, q_imname + '.jpg'), grid_img)

    return cmc[0], cmc[4]


