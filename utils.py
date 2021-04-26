import numpy
import torch
import os
import torchvision.utils as vision_utils
import json
import numpy as np
import matplotlib.pyplot as plt
from evals import compute_mu_sigma_pretrained_model, get_metrics
from data import load_mnist
from model import pretrained_mnist_model
import operator


def save_models(G, D, opt_G, opt_D, out_dir, suffix, withoutOpt=False):
    torch.save(G.state_dict(), os.path.join(out_dir, f"gen_{suffix}.pth"))
    torch.save(D.state_dict(), os.path.join(out_dir, f"disc_{suffix}.pth"))
    if not withoutOpt:
        torch.save(opt_G.state_dict(), os.path.join(out_dir, f"gen_optim_{suffix}.pth"))
        torch.save(
            opt_D.state_dict(), os.path.join(out_dir, f"disc_optim_{suffix}.pth")
        )


def get_num_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def detach_tuple(Tuple):
    return tuple(x.detach_() for x in Tuple)


def get_plot_func(out_dir, img_size, num_samples_eval=10000, save_curves=None):
    dataset = load_mnist(_data_root="datasets", binarized=False)
    # shutil.rmtree(out_dir, ignore_errors=True)
    # if not os.path.exists(out_dir):
    #  os.makedirs(out_dir)
    pretrained_clf = pretrained_mnist_model(
        pretrained="./drive/My Drive/Data/models/mnist.pth"
    )
    mu_real, sigma_real = compute_mu_sigma_pretrained_model(dataset, pretrained_clf)
    (
        inception_means,
        inception_stds,
        inception_means_ema,
        inception_means_avg,
        fids,
        fids_ema,
        fids_avg,
    ) = ([], [], [], [], [], [], [])
    iterations, times = [], []

    def plot_func(
        samples, iteration, time_tick, G=None, D=None, G_avg=None, G_ema=None
    ):
        fig = plt.figure(figsize=(12, 5), dpi=100)
        plt.subplot(1, 2, 1)
        samples = samples.view(100, *img_size)
        file_name = os.path.join(out_dir, "%08d.png" % iteration)
        vision_utils.save_image(samples, file_name, nrow=10)
        grid_img = vision_utils.make_grid(samples, nrow=10, normalize=True, padding=0)
        plt.imshow(grid_img.permute(1, 2, 0), interpolation="nearest")
        plt.subplot(1, 2, 2)
        metrics = get_metrics(pretrained_clf, num_samples_eval, mu_real, sigma_real, G)
        fids.append(metrics["fid"])
        inception_means.append(metrics["inception_mean"])
        inception_stds.append(metrics["inception_std"])
        if G_avg is not None:
            metrics = get_metrics(
                pretrained_clf, num_samples_eval, mu_real, sigma_real, G_avg
            )
            fids_avg.append(metrics["fid"])
            inception_means_avg.append(metrics["inception_mean"])
        if G_ema is not None:
            metrics = get_metrics(
                pretrained_clf, num_samples_eval, mu_real, sigma_real, G_ema
            )
            fids_ema.append(metrics["fid"])
            inception_means_ema.append(metrics["inception_mean"])
        iterations.append(iteration)
        times.append(time_tick)
        #  is
        is_low = [m - s for m, s in zip(inception_means, inception_stds)]
        is_high = [m + s for m, s in zip(inception_means, inception_stds)]
        plt.plot(times, inception_means, label="is", color="r")
        plt.fill_between(times, is_low, is_high, facecolor="r", alpha=0.3)
        plt.yticks(np.arange(0, 10 + 1, 0.5))
        # fid
        plt.plot(times, fids, label="fid", color="b")
        plt.xlabel("Time (sec)")
        plt.ylabel("Metric")
        plt.grid()
        ax = fig.gca()
        ax.set_ylim(-0.1, 10)
        plt.legend(fancybox=True, framealpha=0.5)
        curves_img_file_name = os.path.join(out_dir, "curves.png")
        fig.savefig(curves_img_file_name)
        plt.show()
        curves_file_name = os.path.join(out_dir, "curves.json")
        curves = {
            "inception_means": list(inception_means),
            "inception_stds": list(inception_stds),
            "inception_means_ema": list(inception_means_ema),
            "inception_means_avg": list(inception_means_avg),
            "fids_ema": list(fids_ema),
            "fids_avg": list(fids_avg),
            "fids": list(fids),
            "iterations": iterations,
            "times": times,
        }
        with open(curves_file_name, "w") as fs:
            json.dump(curves, fs)

    return plot_func


def get_update_tuple(first, second, third=None, eta=1, isBoth=False, isadpative=False):
    if isadpative:
        assert isinstance(
            eta, (list, tuple)
        ), "eta needs to be list or tuple because it's adaptive"
        assert not isBoth, "adaptive weight is not support for both yet!"
        return tuple(map(operator.add, first, tuple(map(operator.mul, second, eta))))
    else:
        if isBoth:
            return tuple(
                map(
                    operator.add,
                    tuple(
                        map(operator.add, first, tuple(map(lambda x: x * eta, second)))
                    ),
                    tuple(map(lambda x: x * eta, third)),
                )
            )
        else:
            return tuple(
                map(operator.add, first, tuple(map(lambda x: x * eta, second)))
            )


def adpative_weight_func(method="median", alpha=0.1, top=0.01, least_num_entry=10):
    def adpative_weight(
        f, s, method=method, alpha=alpha, top=top, least_num_entry=least_num_entry
    ):
        first = torch.abs(f.detach().reshape(-1))
        second = torch.abs(s.detach().reshape(-1))
        if method == "median":
            return alpha * torch.median(first).item() / torch.median(second).item()
        elif method == "mean":
            return alpha * torch.mean(first).item() / torch.mean(second).item()
        elif method == "top":
            top_idx = int(len(second) * top)
            assert (
                top_idx > least_num_entry
            ), "too few params in the second input, please decrease the top"
            second_topmean = torch.mean(
                (torch.sort(second, descending=True)[0][:top_idx])
            )
            return alpha * torch.median(first).item() / second_topmean.item()

    return adpative_weight
