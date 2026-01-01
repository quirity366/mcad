import os
import time
import random
import argparse
from typing import Tuple, List

import torch
import torchvision
import numpy as np
from torchvision.transforms import transforms, InterpolationMode
from PIL import Image as PILImage

from models import VQVAE, build_car
import config


# =========================================================
# Utils
# =========================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_torch(tf32=True):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =========================================================
# Model
# =========================================================

def load_models(
    vae_ckpt: str,
    car_ckpt: str,
    device: torch.device,
    model_depth: int,
    patch_nums: Tuple[int],
):
    vae, car = build_car(
        device=device,
        patch_nums=patch_nums,
        depth=model_depth,
        shared_aln=False,
    )

    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"))
    car.load_state_dict(
        torch.load(car_ckpt, map_location="cpu")["trainer"]["car_wo_ddp"]
    )

    vae.eval()
    car.eval()
    for p in list(vae.parameters()) + list(car.parameters()):
        p.requires_grad_(False)

    return vae, car


# =========================================================
# Preprocess
# =========================================================

def preprocess_single_image(
    image_path: str,
    condition_path: str,
    mask_path: str,
    patch_nums: Tuple[int],
    image_size: int,
    device: torch.device,
):
    def pil_loader(path):
        with open(path, "rb") as f:
            return PILImage.open(f).convert("RGB")

    def get_control_for_each_scale(control_image, scale):
        res = []
        for pn in scale:
            res.append(control_image.resize((pn * 16, pn * 16)))
        return res
    
    def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
        return x.add(x).add_(-1)

    # ===== Load =====
    sample = pil_loader(image_path)
    good = sample.copy()
    condition_sample = pil_loader(condition_path)
    mask_sample = pil_loader(mask_path)

    # ===== Resize (infer mode) =====
    resize = transforms.Resize(
        (image_size, image_size),
        interpolation=InterpolationMode.LANCZOS,
    )

    sample = resize(sample)
    condition_sample = resize(condition_sample)

    # ===== Control images =====
    control_images = get_control_for_each_scale(condition_sample, patch_nums)

    # ===== Post transforms =====
    post_trans = transforms.Compose([
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ])

    sample = post_trans(sample)
    control_tensors = [post_trans(ci) for ci in control_images]

    mask_tensor = transforms.ToTensor()(mask_sample)
    good_tensor = transforms.ToTensor()(good)

    # ===== Add batch dim =====
    sample = sample.unsqueeze(0).to(device)
    control_tensors = [c.unsqueeze(0).to(device) for c in control_tensors]
    mask_tensor = mask_tensor.unsqueeze(0).to(device)
    good_tensor = good_tensor.unsqueeze(0).to(device)

    return good_tensor, sample, control_tensors, mask_tensor


# =========================================================
# Inference
# =========================================================

@torch.inference_mode()
def run_single_inference(
    car,
    image,
    controls,
    mask,
    label: int,
    seed: int,
    cfg: float,
    device: torch.device,
):
    label_tensor = torch.tensor([label], device=device)

    images = car.car_inference(
        B=1,
        label_B=label_tensor,
        cfg=cfg,
        top_k=900,
        top_p=0.95,
        g_seed=seed,
        more_smooth=False,
        control_tensors=controls,
        good_B3HW=image,
        mask_B3HW=mask,
    )

    return images


# =========================================================
# Save
# =========================================================

def save_result(
    image,
    save_root: str,
    name: str,
):
    os.makedirs(save_root, exist_ok=True)

    img = image[0].permute(1, 2, 0).mul(255).byte().cpu().numpy()

    PILImage.fromarray(img).save(os.path.join(save_root, f"{name}.png"))


# =========================================================
# Main
# =========================================================

def main(args):
    device = get_device()
    set_seed(args.seed)
    setup_torch(tf32=True)

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    _, car = load_models(
        args.vae_ckpt,
        args.car_ckpt,
        device,
        model_depth=args.model_depth,
        patch_nums=patch_nums,
    )

    good, image, controls, mask = preprocess_single_image(
        image_path=args.image_path,
        condition_path=args.condition_path,
        mask_path=args.mask_path,
        patch_nums=patch_nums,
        image_size=256,
        device=device,
    )

    images = run_single_inference(
        car,
        image=image,
        controls=controls,
        mask=mask,
        label=args.label,
        seed=args.seed,
        cfg=args.cfg,
        device=device,
    )

    save_result(
        images,
        save_root=args.save_root,
        name="result",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CAR Single Image Inference")

    parser.add_argument("--vae_ckpt", default=config.VAE_CKPT)
    parser.add_argument("--car_ckpt", default=config.CAR_CKPT)

    parser.add_argument("--image_path", default='good.png')
    parser.add_argument("--condition_path", default='condition.png')
    parser.add_argument("--mask_path", default='mask.png')

    parser.add_argument("--label", type=int, default=2)

    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--model_depth", type=int, default=16)

    parser.add_argument("--save_root", default=".")

    args = parser.parse_args()

    main(args)
