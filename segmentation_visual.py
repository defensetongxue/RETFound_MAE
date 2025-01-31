# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import argparse
import numpy as np
import os,json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import timm

assert timm.__version__ == "0.3.2" # version check

import util.lr_decay as lrd
from util.datasets import ridge_visual_dataset
from util.pos_embed import interpolate_pos_embed

import models_vit
from models_vit import visual_heatmap



def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--data_path', default='../autodl-tmp/dataset_ROP', type=str,
                        help='dataset path')
    parser.add_argument('--split_name', default='1', type=str,
                        help='which split to load')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')

    # * Finetuning params
    parser.add_argument('--model_path', default='./finetune_rop/checkpoint-best.pth',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # Dataset parameters
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')

    parser.add_argument('--save_dir', default='./experiments/visual',
                        help='path where to save, empty for no saving')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)


    return parser


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    dataset_test = ridge_visual_dataset(
        data_path=args.data_path, split='test', split_name=args.split_name)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False
    )
    model = models_vit.__dict__[args.model](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    checkpoint = torch.load(args.model_path, map_location='cpu')
    checkpoint_model = checkpoint['model']
    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)
    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False) 
    print(msg)
    model.to(device)
    model.eval()
    with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
        data_dict=json.load(f)
    for x,labels,image_names in data_loader_test:
        x=x.cuda()
        outputs=model(x).cpu()
        outputs=torch.softmax(outputs,dim=1)
        predicts=torch.argmax(outputs,dim=1).detach().numpy()
        att_map=model._get_attention_map(x).cpu().numpy()
        for att,label,pred,image_name in zip(att_map,labels,predicts,image_names):
            print(int(label),pred)
            if label != pred:
                visual_heatmap(
                image_path=data_dict[image_name]['image_path'],
                attention_heatmap=att,
                save_path=os.path.join(args.save_dir,image_name)
            )
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        os.system(f"rm -rf {args.save_dir}/*")
    main(args)