import argparse
from datetime import datetime
from pathlib import Path

from train import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ECG Generation via DDIM")

    # Multi Head train params:
    # If set to false, a path to the trained denoising model needs
    # to be provided
    parser.add_argument('-td', '--train-ddim',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-dp', '--ddim-path', type=str, default="None")

    # * General ML Hyperparams:
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-4)
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('-ema', '--ema-momentum', type=float, default=0.990)
    parser.add_argument('-lf', '--loss-function',
                        type=str, default="mean_abs_error")
    # The seed used for all JAX PRNGKeys
    parser.add_argument('-s', '--seed', type=int, default=0)

    # * Network specific Params:
    # The Method used to normalize all input going into the network.
    # Use "Identity" for not normalizing input, "BatchNorm" for BatchNorm
    parser.add_argument('-nrm', '--normalization-method',
                        type=str, default='Identity')
    # The Feature dimension inside the unets. The length of the list
    # also determines the layers of the Unet, where each layer halves the series length
    parser.add_argument('-fs', '--feature-sizes', nargs="+",
                        type=int, default=[32, 64, 96, 128, 160])
    # The amount of ResnetBlocks that each layer of the UNet should have
    parser.add_argument('-bd', '--block-depths', type=int, default=3)
    # The index of the UNet layer at which to start using attention. Set bigger than the UNet depths
    # to not use Attention at all.
    parser.add_argument('-ad', '--attention-depths', type=int, default=2)

    # * Output specific Params:
    parser.add_argument('-pds', '--plot-diffusion-steps', type=int, default=29)
    now = datetime.now().strftime('%Y%m%d-%H%M%S')
    parser.add_argument('-o', '--output-dir', type=str,
                        default=f'./outputs/{now}')
    args = parser.parse_args()

    Path(f"{args.output_dir}/images").mkdir(parents=True, exist_ok=True)
    Path(f"{args.output_dir}/logs").mkdir(parents=True, exist_ok=True)
    Path(f"{args.output_dir}/checkpoints").mkdir(parents=True, exist_ok=True)

    args.log_dir = f"{args.output_dir}/logs"
    args.img_dir = f"{args.output_dir}/images"
    args.ckpt_dir = f"{args.output_dir}/checkpoints"

    train(args)
