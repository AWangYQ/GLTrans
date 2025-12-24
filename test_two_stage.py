import os
from config import cfg
import argparse
from datasets import make_dataloader
from model.make_model import make_two_stage_model
from processor.processor_two import do_inference
from utils.logger import setup_logger
import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    runId = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, runId)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    original_model, refined_model = make_two_stage_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    refined_model.load_param('/root/autodl-tmp/outputs/new_idea/2022-12-10_17-58-30/transformer_120.pth')
    original_model.load_param(cfg.TEST.WEIGHT)
    do_inference(cfg,
                original_model,
                refined_model,
             val_loader,
             num_query)

