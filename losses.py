import torch


def get_disciminator_loss(D, x_real, x_gen, lbl_real, lbl_fake):
    """"""
    D_x = D(x_real)
    D_G_z = D(x_gen)
    lossD_real = torch.binary_cross_entropy_with_logits(D_x, lbl_real).mean()
    lossD_fake = torch.binary_cross_entropy_with_logits(D_G_z, lbl_fake).mean()
    lossD = lossD_real + lossD_fake
    return lossD


def get_generator_loss(G, D, z, lbl_real, is_zerosum=False):
    """"""
    D_G_z = D(G(z))
    if is_zerosum:
        lossG = -torch.binary_cross_entropy_with_logits(1 - D_G_z, lbl_real).mean()
    else:
        lossG = torch.binary_cross_entropy_with_logits(D_G_z, lbl_real).mean()
    return lossG
