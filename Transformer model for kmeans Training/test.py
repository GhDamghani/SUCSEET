import numpy as np
from os.path import join, exists
from os import listdir
from model import TransformerModel
import torch
from model import TransformerModel
import matplotlib.pyplot as plt
import torch.nn.functional as F

num_classes = 20
w = 100

test_data_path = 'test_data'


def get_data(filename):
    while not (exists(filename)):
        pass
    read_flag = False
    while not (read_flag):
        try:
            with np.load(filename) as z:
                y = z['y'].flatten().astype('int64')
                X1 = z['X1'].astype('float32')
                X2 = z['X2'].astype('int64')
            read_flag = True
        except:
            pass
            # print('Error!')

    X1 = torch.tensor(X1).to(device)
    X2 = torch.tensor(X2).to(device)
    X2 = F.one_hot(X2, num_classes).to(torch.float32)
    y = torch.tensor(y).to(device)
    return X1, X2, y
    
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
d_model = 128
d_in = 127
d_out = num_classes
num_heads = d_model//8
dropout = 0.2
num_layers = 4
dim_feedforward = d_model*4

model = TransformerModel(d_in, d_out, d_model, num_heads, dim_feedforward, dropout, num_layers).to(device)
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
    X1, X2, y = get_data(filename)

    pred = model(X1, X2).view(-1, num_classes)

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

print(f'Total Pred Counts: {np.sum(pred_counts)}, Total Pred Labels: {np.sum(pred_label)}')
print(f'Total y Counts: {np.sum(y_counts)}, Total y Labels: {np.sum(y)}')

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