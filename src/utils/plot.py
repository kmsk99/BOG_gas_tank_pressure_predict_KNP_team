import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from utils.utils import *


def epoch_hist(train_hist, valid_hist, name, path="../visualize/"):
    fig = plt.figure(figsize=(10, 4))
    plt.plot(train_hist, label="Training loss")
    plt.plot(valid_hist, label="Valid loss")
    plt.legend()
    plt.savefig(f"{path}{name}_hist.png")


def clustermap(data, path="../visualize/"):
    correlations = data.corr()
    sns_clustermap = sns.clustermap(
        correlations,
        method="complete",
        cmap="RdBu",
        annot=True,
        annot_kws={"size": 14},
        vmin=-1,
        vmax=1,
        figsize=(30, 24),
    )
    sns_clustermap.figure.savefig(f"{path}clustermap_{data.shape[1]}.png")


def plot_two(pred_inverse, validY_inverse, name, path="../visualize/"):
    fig = plt.figure(figsize=(8, 3))
    plt.plot(np.arange(len(pred_inverse)), pred_inverse, label="pred")
    plt.plot(np.arange(len(validY_inverse)), validY_inverse, label="true")
    plt.title("Loss plot")
    plt.savefig(f"{path}{name}_plot.png")


def plot_diff(pred_inverse, validY_inverse, name, path="../visualize/"):
    fig = plt.figure(figsize=(16, 6))
    plt.plot(np.arange(len(pred_inverse)), pred_inverse - validY_inverse, label="pred")
    plt.title("Loss plot")
    plt.savefig(f"{path}{name}_plot_diff.png")
