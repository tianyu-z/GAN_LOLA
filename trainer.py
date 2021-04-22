import torch
import torch.nn as nn
import time
import copy
from data import get_sampler
from updates import Lookahead, update_avg_gen, update_ema_gen
from losses import get_generator_loss, get_disciminator_loss
from utils import save_models


def train(
    G,
    D,
    dataset,
    iterations,
    batch_size=32,
    lrD=0.01,
    lrG=0.01,
    beta1=0.99,
    eval_every=100,
    n_workers=5,
    device=torch.device("cpu"),
    grad_max_norm=1,
    plot_func=lambda a, b, c, d: None,
    extragrad=False,
    lookahead=False,
    lookahead_k=5,
    eval_avg=False,
    out_dir=None,
):

    sampler = get_sampler(
        dataset, batch_size, shuffle=True, num_workers=n_workers, drop_last=True
    )

    if extragrad:
        D_extra = copy.deepcopy(D)
        G_extra = copy.deepcopy(G)
    else:
        D_extra = D
        G_extra = G

    # Optimizers
    optimizerD = torch.optim.Adam(D.parameters(), lr=lrD, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=lrG, betas=(beta1, 0.999))
    if lookahead:
        optimizerD = Lookahead(optimizerD, k=lookahead_k)
        optimizerG = Lookahead(optimizerG, k=lookahead_k)

    optimizerD_extra = torch.optim.Adam(
        D_extra.parameters(), lr=lrD, betas=(beta1, 0.999)
    )
    optimizerG_extra = torch.optim.Adam(
        G_extra.parameters(), lr=lrG, betas=(beta1, 0.999)
    )

    # LBLs
    lbl_real = torch.ones(batch_size, 1, device=device)
    lbl_fake = torch.zeros(batch_size, 1, device=device)

    fixed_noise = torch.randn(100, G.noise_dim, device=device)

    G.to(device)
    D.to(device)

    G_extra.to(device)
    D_extra.to(device)

    G_avg, G_ema = None, None
    if eval_avg:
        G_avg = copy.deepcopy(G)
        G_ema = copy.deepcopy(G)

    start_time = time.perf_counter()

    for i in range(iterations):

        # STEP 1: get G_{t+1} (G_extra)
        if extragrad:
            optimizerG_extra.zero_grad()
            z = torch.randn(batch_size, G_extra.noise_dim, device=device)
            lossG = get_generator_loss(G_extra, D, z, lbl_real)
            lossG.backward()
            optimizerG_extra.step()

        # STEP 2: Get D_{t+1} (D_extra)
        if extragrad:
            optimizerD_extra.zero_grad()
            x_real, _ = sampler()
            x_real = x_real.to(device)
            z = torch.randn(batch_size, G.noise_dim, device=device)
            with torch.no_grad():
                x_gen = G(z)
            lossD = get_disciminator_loss(D_extra, x_real, x_gen, lbl_real, lbl_fake)
            lossD.backward()
            optimizerD_extra.step()

        # STEP 3: D optimization step using G_extra
        x_real, _ = sampler()
        x_real = x_real.to(device)
        z = torch.randn(batch_size, G.noise_dim, device=device)
        with torch.no_grad():
            x_gen = G_extra(z)  # using G_{t+1}
        optimizerD.zero_grad()
        lossD = get_disciminator_loss(D, x_real, x_gen, lbl_real, lbl_fake)
        lossD.backward()
        if grad_max_norm is not None:
            nn.utils.clip_grad_norm_(D.parameters(), grad_max_norm)
        optimizerD.step()

        # STEP 4: G optimization step using D_extra
        z = torch.randn(batch_size, G.noise_dim, device=device)
        optimizerG.zero_grad()
        lossG = get_generator_loss(G, D_extra, z, lbl_real)  # we use the unrolled D
        lossG.backward()
        if grad_max_norm is not None:
            nn.utils.clip_grad_norm_(G.parameters(), grad_max_norm)
        optimizerG.step()

        if extragrad:
            G_extra.load_state_dict(G.state_dict())
            D_extra.load_state_dict(D.state_dict())

        if eval_avg:
            update_avg_gen(G, G_avg, i)
            update_ema_gen(G, G_ema, beta_ema=0.9999)

        if lookahead and (i + 1) % lookahead_k == 0:
            optimizerG.update_lookahead()
            optimizerD.update_lookahead()

        if i % 20000 == 0:
            save_models(G, D, optimizerG, optimizerD, out_dir, suffix=f"{i}")

        # Just plotting things
        if i % eval_every == 0 or i == iterations - 1:
            if out_dir is not None:
                save_models(G, D, optimizerG, optimizerD, out_dir, suffix="last")
            with torch.no_grad():
                probas = torch.sigmoid(D(G(fixed_noise)))
                mean_proba = probas.mean().cpu().item()
                std_proba = probas.std().cpu().item()
                samples = G(fixed_noise)
            print(
                f"Iter {i}: Mean proba from D(G(z)): {mean_proba:.4f} +/- {std_proba:.4f}"
            )
            plot_func(
                samples.detach().cpu(),
                time_tick=time.perf_counter() - start_time,
                D=D,
                G=G,
                iteration=i,
                G_avg=G_avg,
                G_ema=G_ema,
            )
