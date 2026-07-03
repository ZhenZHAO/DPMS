import argparse
import logging
import os
import random
import shutil
import sys
import yaml

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.mixaugs import cut_mix_3d
from dataloaders.LA import (LAHeart, CenterCrop, RandomCrop, WeakStrongAugment,
                            RandomRotFlip, ToTensor, TwoStreamBatchSampler)
from dataloaders.LA import Pancreas
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from utils.util import update_values, set_device
from val_3D import var_all_case_LA, var_all_case_Pancrease


def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args["consistency"] * ramps.sigmoid_rampup(epoch, args["consistency_rampup"])


def update_ema_variables(model, ema_model, alpha, global_step, args):
    # adjust the momentum param
    if global_step < args["consistency_rampup"]:
        alpha = 0.0 
    else:
        alpha = min(1 - 1 / (global_step - args["consistency_rampup"] + 1), alpha)
    
    # update weights
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    
    # update buffers
    if args["flag_ema_buffer"]:
        for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
            buffer_eval.data = buffer_eval.data * alpha + buffer_train.data * (1 - alpha)
    else:
        for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
            buffer_eval.data = buffer_train.data * 1.0


def train(args, snapshot_path):
    base_lr = args["base_lr"]
    batch_size = args["batch_size"]
    max_iterations = args["max_iterations"]
    num_classes = args["num_classes"]
    writer = SummaryWriter(snapshot_path + '/log')

    def create_model(ema=False):
        # Network definition
        net = net_factory(net_type=args["model"], in_chns=1, class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    
    def worker_init_fn(worker_id):
        random.seed(args["seed"] + worker_id)

    # + + + + + + + + + + + #
    # 1. create model
    # + + + + + + + + + + + #
    model = create_model()
    ema_model = create_model(ema=True)
    model.train()
    if args["flag_ema_mode_train"]:
        ema_model.train()
    else:
        ema_model.eval()

    # + + + + + + + + + + + #
    # 2. dataset
    # + + + + + + + + + + + #
    # TODO: will be updated upon accecptance.
    labeled_idxs = list(range(0, args["labeled_num"]))
    unlabeled_idxs = list(range(args["labeled_num"], args["max_samples"]))

    if args.get("flag_sampling_based_on_lb", False):
        batch_sampler = TwoStreamBatchSampler(
            labeled_idxs, unlabeled_idxs, batch_size, batch_size-args["labeled_bs"])
    else:
        batch_sampler = TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, batch_size, args["labeled_bs"])

    # + + + + + + + + + + + #
    # 3. dataloader
    # + + + + + + + + + + + #
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=args["workers"], pin_memory=True, worker_init_fn=worker_init_fn)

    logging.info("{} iterations per epoch".format(len(trainloader)))
    
    # + + + + + + + + + + + #
    # 4. optim, scheduler
    # + + + + + + + + + + + #
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    # + + + + + + + + + + + #
    # 5. training loop
    # + + + + + + + + + + + #
    # TODO: will be updated upon accecptance.
    return "Training Finished!"


if __name__ == "__main__":
    # 1. set up config
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='',
                        help='configuration file')

    # Basics: Data, results, model
    parser.add_argument('--root_path', type=str,
                        default='./data/LA', help='Name of Experiment')
    parser.add_argument('--res_path', type=str, 
                        default='./results/LA', help='Path to save resutls')
    parser.add_argument('--exp', type=str,
                        default='LA/POST', help='experiment_name')
    parser.add_argument('--model', type=str,
                        default='vnet', help='model_name')
    parser.add_argument('--num_classes', type=int,  default=2,
                        help='output channel of network')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='the id of gpu used to train the model')
    
    # Training Basics
    parser.add_argument('--max_iterations', type=int,
                        default=15000, help='maximum epoch number to train')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    # parser.add_argument('--patch_size', type=list,  default=[112, 112, 80],
    #                     help='patch size of network input')
    # https://blog.csdn.net/qq_43391414/article/details/122992458
    parser.add_argument('--patch_size', type=int, nargs="+", default=[112, 112, 80],
                        help='patch size of network input')

    parser.add_argument('--max_samples', type=int,  
                        default=80, help='maximum samples to train')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=2023, help='random seed')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--test_interval_iter', type=int,
                        default=200, help='')
    parser.add_argument('--save_interval_epoch', type=int,
                        default=1000000, help='')
    parser.add_argument("-p", "--poly", default=False, 
                        action='store_true', help="whether poly scheduler")

    # label and unlabel
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size per gpu')
    parser.add_argument('--labeled_bs', type=int, default=2,
                        help='labeled_batch_size per gpu')
    parser.add_argument('--labeled_num', type=int, default=4,
                        help='labeled data')
    
    # model related
    parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
    parser.add_argument("--flag_extra_weak", default=False, 
                        action='store_true', help="whether further apply weak")
    parser.add_argument("--flag_pseudo_from_student", default=False, 
                        action='store_true', help="using pseudo from student itself")
    parser.add_argument("--flag_ema_buffer", default=False, 
                        action='store_true', help="using the ema buffer for teacher")
    parser.add_argument("--flag_ema_mode_train", default=False, 
                        action='store_true', help="using the train model for ema teacher")
    parser.add_argument("--flag_use_teacher_warmup", default=False, 
                        action='store_true', help="using student itself to warmup teacher")

    # augmentation
    parser.add_argument('--cutmix_prob', type=float,  
                        default=0.2, help='probability of applying cutmix')
    parser.add_argument("--flag_use_dynamic_cutmix", default=False, 
                        action='store_true', help="using dynamic cutmix")
    parser.add_argument("--flag_use_color_augs", 
                        default=False, action='store_true', help="using dynamic cutmix")
    parser.add_argument('--add_noise_prob', type=float,  
                        default=1.0, help='probability of adding noise')
    parser.add_argument("--flag_not_use_noise_weak", 
                        default=False, action='store_true', help="using dynamic cutmix")

    # unlabeled loss
    parser.add_argument('--consistency', type=float,
                        default=1.0, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                        default=100.0, help='consistency_rampup')
    parser.add_argument("--conf_threshold", 
                        type=float,
                        default=0.95, 
                        help="confidence threshold for using pseudo-labels")
    parser.add_argument('--flag_ulb_loss_type', type=str,
                        default="dice", help='loss type, ce, dice, dice+ce')

    parser.add_argument("--flag_sampling_based_on_lb", 
                        default=False, action='store_true', help="using dynamic cutmix")

    # parse args
    args = parser.parse_args()
    args = vars(args)

    # 2. update from the config files
    cfgs_file = args['cfg']
    cfgs_file = os.path.join('./cfgs',cfgs_file)
    with open(cfgs_file, 'r') as handle:
        options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
    # convert "1e-x" to float
    for each in options_yaml.keys():
        tmp_var = options_yaml[each]
        if type(tmp_var) == str and "1e-" in tmp_var:
            options_yaml[each] = float(tmp_var)
    # update original parameters of argparse
    update_values(options_yaml, args)
    import pprint

    # 3. setup gpus and randomness
    if args["gpu_id"] in range(8):
        gid = args["gpu_id"]
    else:
        gid = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)

    if not args["deterministic"]:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args["seed"] > 0:
        random.seed(args["seed"])
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        torch.cuda.manual_seed(args["seed"])

    # 4. outputs and logger
    snapshot_path = "{}/{}_{}_labeled/{}".format(
        args["res_path"], args["exp"], args["labeled_num"], args["model"])
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    # if os.path.exists(snapshot_path + '/post'):
    #     shutil.rmtree(snapshot_path + '/post')
    # shutil.copytree('.', snapshot_path + '/post',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{}".format(pprint.pformat(args)))

    train(args, snapshot_path)
