import numpy
import torch
import torch.nn as nn
from collections import OrderedDict
import os


class DiscriminatorCNN28(nn.Module):
    def __init__(
        self,
        img_channels=1,
        h_filters=64,
        spectral_norm=False,
        img_size=None,
        n_outputs=1,
        affine=False,
    ):
        if any(
            not isinstance(_arg, int) for _arg in [img_channels, h_filters, n_outputs]
        ):
            raise TypeError("Unsupported operand type. Expected integer.")
        if not isinstance(spectral_norm, bool):
            raise TypeError(
                f"Unsupported operand type: {type(spectral_norm)}. " "Expected bool."
            )
        if min([img_channels, h_filters, n_outputs]) <= 0:
            raise ValueError(
                "Expected nonzero positive input arguments for: the "
                "number of output channels, the dimension of the noise "
                "vector, as well as the depth of the convolution kernels."
            )
        super(DiscriminatorCNN28, self).__init__()
        # _conv = nn.utils.spectral_norm(nn.Conv2d) if spectral_norm else nn.Conv2d
        _apply_sn = lambda x: nn.utils.spectral_norm(x) if spectral_norm else x
        self.img_channels = img_channels
        self.img_size = img_size
        self.n_outputs = n_outputs
        self.main = nn.Sequential(
            _apply_sn(nn.Conv2d(img_channels, h_filters, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            _apply_sn(nn.Conv2d(h_filters, h_filters * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(h_filters * 2, affine=affine),
            nn.LeakyReLU(0.2, inplace=True),
            _apply_sn(nn.Conv2d(h_filters * 2, h_filters * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(h_filters * 4, affine=affine),
            nn.LeakyReLU(0.2, inplace=True),
            _apply_sn(nn.Conv2d(h_filters * 4, self.n_outputs, 3, 1, 0, bias=False)),
        )

    def forward(self, x):
        if self.img_channels is not None and self.img_size is not None:
            if (
                numpy.prod(list(x.size())) % (self.img_size ** 2 * self.img_channels)
                != 0
            ):
                raise ValueError(
                    f"Size mismatch. Input size: {numpy.prod(list(x.size()))}. "
                    f"Expected input divisible by: {self.noise_dim}"
                )
            x = x.view(-1, self.img_channels, self.img_size, self.img_size)
        x = self.main(x)
        return x.view(-1, self.n_outputs)

    def load(self, model):
        self.load_state_dict(model.state_dict())


class GeneratorCNN28(nn.Module):
    def __init__(
        self, img_channels=1, noise_dim=128, h_filters=64, out_tanh=False, affine=False
    ):
        if any(
            not isinstance(_arg, int) for _arg in [img_channels, noise_dim, h_filters]
        ):
            raise TypeError("Unsupported operand type. Expected integer.")
        if min([img_channels, noise_dim, h_filters]) <= 0:
            raise ValueError(
                "Expected strictly positive input arguments for the "
                "number of output channels, the dimension of the noise "
                "vector, as well as the depth of the convolution kernels."
            )
        super(GeneratorCNN28, self).__init__()
        self.noise_dim = noise_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, h_filters * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(h_filters * 8, affine=affine),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(h_filters * 8, h_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(h_filters * 4, affine=affine),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(h_filters * 4, h_filters * 2, 4, 2, 0, bias=False),
            nn.BatchNorm2d(h_filters * 2, affine=affine),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(h_filters * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh() if out_tanh else nn.Sigmoid(),
        )

    def forward(self, x):

        if numpy.prod(list(x.size())) % self.noise_dim != 0:
            raise ValueError(
                f"Size mismatch. Input size: {numpy.prod(list(x.size()))}. "
                f"Expected input divisible by: {self.noise_dim}"
            )
        x = x.view(-1, self.noise_dim, 1, 1)
        x = self.main(x)
        return x

    def load(self, model):
        self.load_state_dict(model.state_dict())


class MLP_mnist(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(MLP_mnist, self).__init__()
        assert isinstance(input_dims, int), "Expected int for input_dims"
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers["fc{}".format(i + 1)] = nn.Linear(current_dims, n_hidden)
            layers["relu{}".format(i + 1)] = nn.ReLU()
            layers["drop{}".format(i + 1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers["out"] = nn.Linear(current_dims, n_class)
        self.layers = layers
        self.model = nn.Sequential(layers)
        # print(self.model)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.model.forward(input)

    def get_logits_and_fc2_outputs(self, x):
        x = x.view(x.size(0), -1)
        assert x.size(1) == self.input_dims
        fc2_out = None
        for l in self.model:
            x = l(x)
            if l == self.layers["fc2"]:
                fc2_out = x
        return x, fc2_out


def pretrained_mnist_model(
    input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=None
):
    model = MLP_mnist(input_dims, n_hiddens, n_class)
    if pretrained is not None:
        if os.path.exists(pretrained):
            print("Loading trained model from %s" % pretrained)
            state_dict = torch.load(
                pretrained,
                map_location="cuda:0" if torch.cuda.is_available() else "cpu",
            )
            if "parallel" in pretrained:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                state_dict = new_state_dict
        else:
            raise FileNotFoundError(f"Could not find pretrained model: {pretrained}.")
        model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model = model.cuda()
    return model
