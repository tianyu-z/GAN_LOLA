import torch
import os
import json
import shutil

from data import load_mnist
from model import DiscriminatorCNN28, GeneratorCNN28
from trainer import train_2nd_order_manual
from utils import get_plot_func

_NOISE_DIM = 16
_H_FILTERS = 8

args = dict(
    iterations=100000,
    batch_size=128,
    lrD=0.001,
    lrG=0.001,
    eta=1,
    eval_every=5000,
    n_workers=4,
    device="cuda",
    type_="lookahead",
)

for k in range(1, 5 + 1):
    exp_key = (
        f"type_{args['type_']}_eta{args['eta']}_iter{args['iterations']}_bs{args['batch_size']}_lrD{args['lrD']}"
        + f"_lrG{args['lrG']}"
        + f"_ee{args['eval_every']}"
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
        eta=args["eta"],
        eval_every=args["eval_every"],
        n_workers=args["n_workers"],
        device=torch.device(args["device"]),
        plot_func=plot_func,
        out_dir=out_dir,
        type_=args["type_"],
    )
