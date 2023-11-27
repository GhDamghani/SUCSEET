import sys
import numpy as np
from os.path import join, exists
from os import listdir, remove, mkdir
import joblib
from sklearn.utils import gen_batches
from multiprocessing import Process, freeze_support
from model import TransformerModel
import torch
from tqdm import tqdm
import shutil
from model import TransformerModel
import matplotlib.pyplot as plt

num_classes = 20
w = 100

test_data_path = 'test_data'


if __name__ == '__main__':
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    d_model = 128
    d_out = num_classes
    num_heads = 8
    dropout = 0.2
    num_layers = 2
    model = TransformerModel(d_model, d_out,
                             dropout=dropout, num_layers=num_layers, num_heads=num_heads).to(device)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    test_loss = 0
    corrects = 0
    total = 0
    y_numpy = []
    pred_numpy = []
    for batch_i in range(len(listdir(test_data_path))):
        filename = join(test_data_path, f'test_B{batch_i:04}.npz')
        while not (exists(filename)):
            pass
        read_flag = False
        while not (read_flag):
            try:
                with np.load(filename) as z:
                    y = z['y'].flatten().astype('int64')
                    X = z['X'].astype('float32')
                read_flag = True
            except:
                pass
                # print('Error!')
        X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

        pred = model(X).view(-1, num_classes)

        pred_label = torch.argmax(pred, -1)
        corrects += torch.sum(y == pred_label).item()
        total += y.numel()
        pred_label = pred_label.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        pred_numpy.append(pred_label)
        y_numpy.append(y)
    
    y = np.concatenate(y_numpy, 0)
    pred_label = np.concatenate(pred_numpy, 0)

    for i in np.random.choice(range(len(y)), 10):
        print(f'True Label: {y[i]}, Predicted: {pred_label[i]}')
    
    print(f'Accuracy: {100*corrects/total:02.3f}%')

    
    
    pred_labels, pred_counts = np.unique(pred_label, return_counts=True)
    y_labels, y_counts = np.unique(y, return_counts=True)

    plt.subplot(2, 1, 1)
    plt.stem(pred_labels, pred_counts, linefmt='-', markerfmt='o', basefmt='k-')
    plt.xticks(range(num_classes))
    plt.xlabel('Labels')
    plt.ylabel('Freq')
    plt.title('Histogram of Predicted Output for test data')
    
    plt.subplot(2, 1, 2)
    plt.stem(y_labels, y_counts, linefmt='-', markerfmt='o', basefmt='k-')
    plt.xticks(range(num_classes))
    plt.xlabel('Labels')
    plt.ylabel('Freq')
    plt.title('Histogram of True Output for test data')

    plt.tight_layout()
    plt.show()
