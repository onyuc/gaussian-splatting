#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import sys

sys.path.append("/scratch/rchkl2380/Workspace/gaussian-splatting")
from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):

                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                psnrs = []
                ssims_1 = []
                ssims_2 = []
                lpipss_vgg = []
                lpipss_alex = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    ssims_1.append(ssim(renders[idx], gts[idx], 1))
                    ssims_2.append(ssim(renders[idx], gts[idx], 2))
                    lpipss_vgg.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    lpipss_alex.append(lpips(renders[idx], gts[idx], net_type='alex'))

                # print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  DSSIM_1 : {:>12.7f}".format((1.0-torch.tensor(ssims_1).mean())/2.0, ".5"))
                print("  DSSIM_2 : {:>12.7f}".format((1.0-torch.tensor(ssims_2).mean())/2.0, ".5"))
                print("  LPIPS_vgg: {:>12.7f}".format(torch.tensor(lpipss_vgg).mean(), ".5"))
                print("  LPIPS_alex: {:>12.7f}".format(torch.tensor(lpipss_alex).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({
                    # "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "DSSIM_1": (1.0 - torch.tensor(ssims_1).mean().item()) / 2.0,
                    "DSSIM_2": (1.0 - torch.tensor(ssims_2).mean().item()) / 2.0,
                    "LPIPS_vgg": torch.tensor(lpipss_vgg).mean().item(),
                    "LPIPS_alex": torch.tensor(lpipss_alex).mean().item(),
                })
                per_view_dict[scene_dir][method].update({
                    # "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "DSSIM_1": {name: (1.0-ssim)/2.0 for ssim, name in zip(torch.tensor(ssims_1).tolist(), image_names)},
                    "DSSIM_2": {name: (1.0-ssim)/2.0 for ssim, name in zip(torch.tensor(ssims_2).tolist(), image_names)},
                    "LPIPS_vgg": {name: lp for lp, name in zip(torch.tensor(lpipss_vgg).tolist(), image_names)},
                    "LPIPS_alex": {name: lp for lp, name in zip(torch.tensor(lpipss_alex).tolist(), image_names)},
                })

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
