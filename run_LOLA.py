import torch
import os
import json
import shutil

from data import load_mnist
from model import DiscriminatorCNN28, GeneratorCNN28
from trainer import train_2nd_order_manual
from utils import get_plot_func
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-noise", "--noise_dim", type=int, default=8, help="noise_dim.",
)
parser.add_argument(
    "-h", "--h_filter", type=int, default=4, help="h_filter.",
)

parser.add_argument(
    "-bs", "--batch_size", type=int, default=128, help="batch size.",
)
parser.add_argument(
    "-iter", "--iterations", type=int, default=300000, help="iterations",
)
parser.add_argument(
    "-lD", "--lrD", type=float, default=0.001, help="lr for D",
)
parser.add_argument(
    "-lG", "--lrG", type=float, default=0.001, help="lr for D",
)
parser.add_argument(
    "-ee", "--eval_every", type=int, default=5000, help="save every iter",
)
parser.add_argument(
    "-nw", "--n_workers", type=int, default=4, help="num of work",
)
parser.add_argument(
    "-d", "--device", type=str, default="cuda", help="device",
)
parser.add_argument(
    "-t", "--type_", type=str, default="lola", help="lola or lookahead",
)
parser.add_argument(
    "--adaptive_weight_opt",
    type=list,
    default=["top", 1, 0.2, 10],
    help="method,alpha,top,least_num_params",
)
args = parser.parse_args()
_NOISE_DIM = args.noise_dim
_H_FILTERS = args.h_filter

for k in range(1, 5 + 1):
    if args["adaptive_weight_opt"] is not None:
        adaptive_weight_opt_name = (
            "_"
            + args["adaptive_weight_opt"][0]
            + "_"
            + str(args["adaptive_weight_opt"][1])
            + "_"
            + str(args["adaptive_weight_opt"][2])
            + "_"
            + str(args["adaptive_weight_opt"][3])
        )
    else:
        adaptive_weight_opt_name = "_"
    exp_key = (
        f"type_{args['type_']}_iter{args['iterations']}_bs{args['batch_size']}_lrD{args['lrD']}"
        + f"_lrG{args['lrG']}"
        + f"_ee{args['eval_every']}"
        + adaptive_weight_opt_name
    )
    out_dir = f"./results/final/{exp_key}/{k}/"

    shutil.rmtree(out_dir, ignore_errors=True)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, "args.json"), "w") as fs:
        json.dump(args, fs)

    dataset = load_mnist(_data_root="datasets", binarized=False)

    plot_func = get_plot_func(
        out_dir=out_dir, img_size=dataset[0][0].size(), num_samples_eval=10000
    )

    G = GeneratorCNN28(noise_dim=_NOISE_DIM, h_filters=_H_FILTERS, out_tanh=True)
    D = DiscriminatorCNN28(h_filters=_H_FILTERS, spectral_norm=False, img_size=28)

    train_2nd_order_manual(
        G,
        D,
        dataset,
        iterations=args["iterations"],
        batch_size=args["batch_size"],
        lrD=args["lrD"],
        lrG=args["lrG"],
        eval_every=args["eval_every"],
        n_workers=args["n_workers"],
        device=torch.device(args["device"]),
        plot_func=plot_func,
        out_dir=out_dir,
        type_=args["type_"],
        adaptive_weight_opt=args["adaptive_weight_opt"],
    )
