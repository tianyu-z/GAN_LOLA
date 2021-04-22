import torch
import time
import numpy as np


# https://pytorch.org/docs/stable/autograd.html # https://stackoverflow.com/questions/64024312/how-to-compute-hessian-matrix-for-all-parameters-in-a-network-in-pytorch
# https://github.com/noahgolmant/pytorch-hessian-eigenthings
def get_second_order_grad(grads, xs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()
    grads2 = []
    for j, (grad, x) in enumerate(zip(grads, xs)):
        print("2nd order on ", j, "th layer")
        print(x.size())
        grad = torch.reshape(grad, [-1])
        grads2_tmp = []
        for count, g in enumerate(grad):
            g2 = torch.autograd.grad(g, x, retain_graph=True)[0]
            g2 = torch.reshape(g2, [-1])
            grads2_tmp.append(g2[count].data.cpu().numpy())
        grads2.append(torch.from_numpy(np.reshape(grads2_tmp, x.size())).to(device))
        print("Time used is ", time.time() - start)
    for grad in grads2:  # check size
        print(grad.size())

    return grads2
