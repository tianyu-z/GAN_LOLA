import numpy
import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg


def compute_mu_sigma_pretrained_model(dataset, pretrained_clf):
    dataloader = DataLoader(dataset, batch_size=512, num_workers=2, drop_last=True)
    cuda = next(pretrained_clf.parameters()).is_cuda
    all_fc2_out = []
    pretrained_clf.eval()
    for batch, _ in dataloader:
        with torch.no_grad():
            if cuda:
                batch = batch.cuda()
            _, fc2_out = pretrained_clf.get_logits_and_fc2_outputs(batch)
        all_fc2_out.append(fc2_out.cpu())
    all_fc2_out = torch.cat(all_fc2_out, dim=0).numpy()
    mu_real = np.mean(all_fc2_out, axis=0)
    sigma_real = np.cov(all_fc2_out, rowvar=False)
    return mu_real, sigma_real


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; adding %s to diagonal of cov estimates"
            % eps
        )
        print(msg)
        # warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def _calculate_metrics(
    pretrained_clf,
    G,
    dataset_length,
    mu_real,
    sigma_real,
    n_classes=10,
    batch_size=1024,
):
    cuda = next(pretrained_clf.parameters()).is_cuda
    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Using pretrained clf to get predictions over fake data
    inception_predictions, all_fc2_out, class_probas = [], [], []
    dataloader = DataLoader(
        list(range(dataset_length)), batch_size, num_workers=2, drop_last=True
    )
    pretrained_clf.eval()
    for batch in dataloader:
        with torch.no_grad():
            noise = torch.randn(batch_size, G.noise_dim, device=device)
            probas, fc2_out = pretrained_clf.get_logits_and_fc2_outputs(
                G(noise).view(batch_size, -1)
            )
        all_fc2_out.append(fc2_out.cpu())
        class_probas.append(probas.cpu())
    all_fc2_out = torch.cat(all_fc2_out, dim=0).numpy()
    class_probas = torch.cat(class_probas, dim=0)
    inception_predictions = torch.softmax(class_probas, dim=1).numpy()
    class_probas = class_probas.numpy()
    pred_prob = np.maximum(class_probas, 1e-20 * np.ones_like(class_probas))

    y_vec = 1e-20 * np.ones(
        (len(pred_prob), n_classes), dtype=np.float
    )  # pred label distr
    gnd_vec = 0.1 * np.ones(
        (1, n_classes), dtype=np.float
    )  # gnd label distr, uniform over classes

    for i, label in enumerate(pred_prob):
        y_vec[i, np.argmax(pred_prob[i])] = 1.0
    y_vec = np.sum(y_vec, axis=0, keepdims=True)
    y_vec = y_vec / np.sum(y_vec)

    label_entropy = np.sum(-y_vec * np.log(y_vec)).tolist()
    label_tv = np.true_divide(np.sum(np.abs(y_vec - gnd_vec)), 2).tolist()
    label_l2 = np.sum((y_vec - gnd_vec) ** 2).tolist()

    # --- is ----
    inception_scores = []
    for i in range(n_classes):
        part = inception_predictions[
            (i * inception_predictions.shape[0] // n_classes) : (
                (i + 1) * inception_predictions.shape[0] // n_classes
            ),
            :,
        ]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        inception_scores.append(np.exp(kl))

    mu = np.mean(all_fc2_out, axis=0)
    sigma = np.cov(all_fc2_out, rowvar=False)
    _fid = calculate_frechet_distance(mu, sigma, mu_real, sigma_real)

    return (
        label_entropy,
        label_tv,
        label_l2,
        float(np.mean(inception_scores)),
        float(np.std(inception_scores)),
        _fid,
    )


def get_metrics(pretrained_clf, dataset_length, mu_real, sigma_real, G):
    """Calculates entropy, TV, L2, and inception scores."""
    e, tv, l2, is_m, is_std, fid = _calculate_metrics(
        pretrained_clf, G, dataset_length, mu_real, sigma_real
    )
    m_result = {
        "entropy": e,
        "TV": tv,
        "L2": l2,
        "inception_mean": is_m,
        "inception_std": is_std,
        "fid": fid,
    }
    return m_result
