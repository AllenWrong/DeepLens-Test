import os

import numpy as np
import torch
import random
from sklearn.metrics import r2_score


def save_checkpoint(checkpoint, filename):
    print("==> Saving checkpoint")
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer):
    print("==> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def diff_rate(y_pred, y_ture):
    """
    Args:
        y_pred (list):
        y_ture (list):
    """
    y_pred_arr = np.array(y_pred)
    y_ture_arr = np.array(y_ture)
    diff = np.abs(y_pred_arr - y_ture_arr)
    return np.mean(diff / y_ture_arr)


def create_description(data_dir, rate_list):
    """
    create train and test description
    Args:
        rate_list (list): contains train_rate, val_rate, test_rate will be
            induced.
    """
    assert len(rate_list) == 2
    samples = os.listdir(data_dir)
    random.shuffle(samples)
    train_end = int(len(samples)*rate_list[0])
    val_end = int(len(samples)*(rate_list[0]+rate_list[1]))
    
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    line = "path\n"
    train_line = ""
    val_line = ""
    test_line = ""

    for sample in train_samples:
        train_line += sample + "\n"

    for sample in val_samples:
        val_line += sample + "\n"

    for sample in test_samples:
        test_line += sample + "\n"

    with open("./train description.csv", "w") as f:
        f.write(line+train_line)
    print("train description.csv has been saved.")

    with open("./val description.csv", "w") as f:
        f.write(line+val_line)
    print("val description.csv has been saved.")

    with open("./test description.csv", "w") as f:
        f.write(line+test_line)
    print("test description.csv has been saved.")


def check_accuracy(loader, model, criterion, device, flatten=False):

    model.eval()
    loss_per_step = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for (x, y) in loader:
            # move data to cuda if gpu is available
            x = x.to(device)
            y = y.to(device)

            if flatten:
                x = x.reshape((x.shape[0], -1))

            score = model.forward(x)
            loss = criterion(score.ravel(), y)
            loss_per_step.append(loss.item())
            y_pred.extend(list(score.cpu().numpy()))
            y_true.extend(list(y.cpu().numpy()))
    model.train()

    r2 = r2_score(y_true, y_pred)
    rate = diff_rate(y_pred, y_true)
    return sum(loss_per_step) / len(loss_per_step), r2, rate
