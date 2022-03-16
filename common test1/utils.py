import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt


def save_checkpoint(checkpoint, file):
    print(f"====> Save checkpoint in {file}")
    torch.save(checkpoint, file)


def load_checkpoint(file, model, optimizer):
    print(f"====> Load model from {file}")
    checkpoint = torch.load(file)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def check_accuracy(loader, model, device, need_flatten=False):
    """
    Args:
        loader: train loader or test/val loader
        model: model
        device: device
        need_flatten: if only mlp is used, this need to be true.

    Returns:
        acc: accuracy
        auc: auc score
        y_ture: shape is (n*class_num,).
            In this shape, we can use roc_auc_score or roc_curve conveniently.
        y_pred: shape is (n*class_num,) same as above.
    """
    num_samples = 0
    num_correct_pred = 0
    model.eval()

    y_pred = []
    y_true = []

    with torch.no_grad():
        for (x, y) in loader:
            # move data to cuda if gpu is available
            x = x.to(device)
            y = y.to(device)

            if need_flatten:
                x = x.reshape(x.shape[0], -1)

            score = model.forward(x)
            val, idx = torch.max(score, dim=1)
            num_correct_pred += (y == idx).sum()
            num_samples += x.shape[0]

            y_pred.extend(list(score.cpu().numpy()))
            y_true.extend((list(y.cpu().numpy())))

        acc = num_correct_pred / num_samples
        y_one_hot = OneHotEncoder().fit_transform(np.array(y_true).reshape(-1, 1)).toarray()
        auc = roc_auc_score(
            y_one_hot.ravel(),
            softmax(np.array(y_pred), axis=1).ravel()
        )

    model.train()
    return acc, auc, y_one_hot.ravel(), softmax(np.array(y_pred), axis=1).ravel()


def plot_roc(fpr, tpr, auc):
    """plot roc curve
    Args:
        fpr (np.ndarray): shape should be (n,)
        tpr (np.ndarray): shape should be (n,)
    """
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.legend()
    plt.savefig("./roc.png", dpi=500)
    plt.show()
