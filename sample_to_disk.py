import argparse
import json
import pathlib

import numpy as np
import torch

import transformer_flow
import utils


def main(args):
    dist = utils.Distributed()
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive")
    if args.batch_size <= 0 or args.batch_size % dist.world_size:
        raise ValueError("--batch_size must be a positive multiple of the world size")
    utils.set_random_seed(args.seed + dist.rank)
    num_classes = utils.get_num_classes(args.dataset)
    if num_classes and args.num_samples % num_classes:
        raise ValueError("--num_samples must be divisible by the number of classes")

    model = transformer_flow.Model(
        in_channels=args.channel_size,
        img_size=args.img_size,
        patch_size=args.patch_size,
        channels=args.channels,
        num_blocks=args.blocks,
        layers_per_block=args.layers_per_block,
        nvp=args.nvp,
        num_classes=num_classes,
    ).cuda()
    # Just load directly - TARFlow does not use EMA
    model.load_state_dict(
        torch.load(args.ckpt_file, map_location="cpu", weights_only=True)
    )
    model.eval().requires_grad_(False)

    if dist.rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        images_out = np.lib.format.open_memmap(
            args.output_dir / "samples.npy",
            mode="w+",
            dtype=np.float32,
            shape=(args.num_samples, args.channel_size, args.img_size, args.img_size),
        )
        if num_classes:
            labels_out = np.lib.format.open_memmap(
                args.output_dir / "labels.npy",
                mode="w+",
                dtype=np.int64,
                shape=(args.num_samples,),
            )
    dist.barrier()

    for start in range(0, args.num_samples, args.batch_size):
        count = min(args.batch_size, args.num_samples - start)
        # Equal local sizes are required by all-gather; trim padding after gathering.
        b = (count + dist.world_size - 1) // dist.world_size
        noise = torch.randn(
            b,
            (args.img_size // args.patch_size) ** 2,
            args.channel_size * args.patch_size**2,
            device="cuda",
        )

        if num_classes:
            # Match the rank-ordered all-gather so saved labels cycle evenly.
            y = (torch.arange(b, device="cuda") + start + dist.rank * b) % num_classes
        else:
            y = None

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            samples = model.reverse(
                noise, y, args.cfg, attn_temp=args.attn_temp, annealed_guidance=True
            )
        samples = samples.float()
        # Tweedie's trick
        if args.self_denoising_lr > 0:
            db = max(1, b // 4)
            for j in range(0, b, db):
                x = samples[j : j + db].detach().clone().requires_grad_(True)
                y_ = y[j : j + db] if y is not None else None
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z, _, logdets = model(x, y_)
                    loss = model.get_loss(z, logdets)
                grad = torch.autograd.grad(loss, x)[0]
                # Undo the loss's averaging over batch and image dimensions.
                lr = args.self_denoising_lr * args.noise_std**2 * x.numel()
                with torch.no_grad():
                    samples[j : j + db] = x - lr * grad

        samples = dist.gather_concat(samples.detach())[:count]
        if not torch.isfinite(samples).all():
            raise RuntimeError("Generated non-finite samples")
        if y is not None:
            y = dist.gather_concat(y)[:count]
        if dist.rank == 0:
            images_out[start : start + count] = samples.cpu().numpy()
            if y is not None:
                labels_out[start : start + count] = y.cpu().numpy()
            print(f"Saved {start + count}/{args.num_samples} samples", flush=True)

    if dist.rank == 0:
        images_out.flush()
        if num_classes:
            labels_out.flush()
        metadata = {
            "config": vars(args),
            "layout": "NCHW",
            "dtype": "float32",
            "pixel_scale": "Model space (nominally [-1, 1]), unclipped",
            "annealed_guidance": True,
            "world_size": dist.world_size,
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str) + "\n"
        )
    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export unclipped FP32 samples in NCHW layout."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=pathlib.Path,
        help="New directory for samples.npy, labels.npy, and metadata.json",
    )

    parser.add_argument(
        "--ckpt_file", required=True, type=pathlib.Path, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--dataset",
        default="imagenet",
        type=str,
        choices=["cifar10", "imagenet", "imagenet64", "afhq"],
        help="Name of dataset",
    )
    parser.add_argument("--img_size", default=32, type=int, help="Image size")
    parser.add_argument(
        "--channel_size", default=3, type=int, help="Image channel size"
    )

    parser.add_argument(
        "--patch_size", default=4, type=int, help="Patch size for the model"
    )
    parser.add_argument("--channels", default=512, type=int, help="Model width")
    parser.add_argument(
        "--blocks", default=4, type=int, help="Number of autoregressive flow blocks"
    )
    parser.add_argument(
        "--layers_per_block", default=8, type=int, help="Depth per flow block"
    )
    parser.add_argument(
        "--noise_std", default=0.05, type=float, help="Input noise standard deviation"
    )
    parser.add_argument("--nvp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--seed", default=100, type=int, help="Base seed; each rank adds its rank"
    )

    parser.add_argument(
        "--cfg",
        default=0,
        type=float,
        help="Guidance weight for sampling, 0 is no guidance. For conditional models consider the range in [1, 3]",
    )

    parser.add_argument(
        "--attn_temp",
        default=1.0,
        type=float,
        help="Attention temperature for unconditional guidance, enabled when not 1 (eg, 0.5, 1.5)",
    )
    parser.add_argument(
        "--batch_size", default=1024, type=int, help="Batch size for drawing samples"
    )
    parser.add_argument(
        "--num_samples", default=50000, type=int, help="Number of total samples to draw"
    )
    parser.add_argument(
        "--self_denoising_lr",
        default=1.0,
        type=float,
        help="Learning rate multiplier for denoising, 1 is the theoretical optimal one",
    )

    args = parser.parse_args()

    main(args)
