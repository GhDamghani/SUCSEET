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

kmeans_folder = join('..', 'Kmeans of sounds')
path_input = join(kmeans_folder, 'features')
participant = 'sub-06'

melSpec = np.load(join(path_input, f'{participant}_spec.npy'))
feat = np.load(join(path_input, f'{participant}_feat.npy'))
kmeans = joblib.load(join(kmeans_folder, 'kmeans.joblib'))
num_classes = 20

no_samples = melSpec.shape[0]
w = 100


def train_test_label(no_samples, r=0.25, w=w, p=0.1, seed=1534):
    l_test = np.int32(np.round(no_samples*r))
    assert l_test > w
    test_pick_list = list(range(w, no_samples-w-l_test+1))  # + [0, -1]
    np.random.seed(seed)  # reproducibility
    test_index_clip = np.random.choice(test_pick_list, 1).item()
    train_indices = list(range(0, test_index_clip-w)) + \
        list(range(test_index_clip+l_test, no_samples-w+1))
    test_indices = list(range(test_index_clip, test_index_clip+l_test-w+1))

    if 0 < p < 1:
        # Only a portion of train data to test the general capability of the model at first, optional
        np.random.seed(28)  # reproducibility
        train_indices = np.random.choice(
            train_indices, int(len(train_indices)*p), replace=False)
        test_indices = np.random.choice(
            test_indices, int(len(test_indices)*p), replace=False)

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    return train_indices, test_indices


train_indices, test_indices = train_test_label(no_samples, p=1)

epochs = 1000
batch_size = 16
no_data_train = len(train_indices)
no_data_test = len(test_indices)
train_batches_indices = list(gen_batches(no_data_train, batch_size))
test_batches_indices = list(gen_batches(no_data_test, batch_size*2))
no_train_batches = len(train_batches_indices)
no_test_batches = len(test_batches_indices)

np.random.seed(26)  # reproducibility
train_shuffle_seeds = np.random.randint(1, 100000, epochs)
train_data_path = 'train_data'
test_data_path = 'test_data'


def train_data_gen(epoch_i, path=train_data_path):
    if not exists(path):
        mkdir(path)
    random_indices = np.arange(no_data_train, dtype=int)
    np.random.seed(train_shuffle_seeds[epoch_i])
    np.random.shuffle(random_indices)
    dataname_train_shuffled = train_indices.copy()[random_indices]
    for (i, x) in enumerate(train_batches_indices):
        trainname_batch = dataname_train_shuffled[x]
        x_batch = []
        y_batch = []
        for s_ind in trainname_batch:  # preprocessing
            x1 = np.concatenate((feat[s_ind:s_ind+w], np.zeros((w, 1))), 1)
            # x1 = feat[s_ind:s_ind+w]
            y0 = melSpec[s_ind:s_ind+w]
            y1 = kmeans.predict(y0)
            x_batch.append(x1)
            y_batch.append(y1)

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        while len(listdir(path)) >= no_train_batches//3:
            pass
        np.savez(
            join(path, f'train_E{epoch_i:04}_B{i:04}'), X=x_batch, y=y_batch)


def test_data_gen(path=test_data_path):
    if not exists(path):
        mkdir(path)
    for i, x in enumerate(test_batches_indices):
        trainname_batch = test_indices[x]
        x_batch = []
        y_batch = []
        for s_ind in trainname_batch:  # preprocessing
            x1 = np.concatenate((feat[s_ind:s_ind+w], np.zeros((w, 1))), 1)
            # x1 = feat[s_ind:s_ind+w]
            y0 = melSpec[s_ind:s_ind+w]
            y1 = kmeans.predict(y0)
            x_batch.append(x1)
            y_batch.append(y1)

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        np.savez(join(path, f'test_B{i:04}'), X=x_batch, y=y_batch)


def train_data_gen_proc(epochs):
    for epoch_i in range(epochs):
        prc = Process(target=train_data_gen, args=(epoch_i,))
        prc.start()
        prc.join()
        prc.terminate()


if __name__ == '__main__':
    freeze_support()
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
    print(model)
    print('Total number of trainable parameters:', sum(p.numel()
          for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    lossfn = torch.nn.CrossEntropyLoss()

    create_test_files = not (exists(test_data_path))

    min_test_loss = float('inf')
    last_test_loss = float('inf')
    best_epoch = 0
    progressive_epoch = 0
    patience = 15
    log_interval = int(np.round_(no_train_batches/5))

    train_prc = Process(target=train_data_gen_proc, args=(epochs,))
    train_prc.start()

    for epoch_i in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        corrects = 0
        total = 0
        for batch_i in tqdm(range(no_train_batches), leave=False):
            filename = join(train_data_path,
                            f'train_E{epoch_i:04}_B{batch_i:04}.npz')
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
            current_batch_size = X.shape[0]
            X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

            # Compute prediction error
            pred = model(X).view(-1, num_classes)
            loss = lossfn(pred, y) * current_batch_size

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            train_loss += loss.item()
            # torch.randint(0, num_classes, X.shape[0:2]).to(device)
            pred_label = torch.argmax(pred, -1)
            corrects += torch.sum(y == pred_label).item()
            total += y.numel()

            remove(filename)

            if ((batch_i + 1) % log_interval == 0) or batch_i == no_train_batches:
                current= min((batch_i + 1) * batch_size, no_data_train)
                accuracy = corrects/total
                lr = scheduler.get_last_lr()[0]
                tqdm.write(
                    f"| Epoch: {epoch_i:04d} | lr: {lr:08.6f} | loss: {train_loss*batch_size/total:03.6f} | acc: [{corrects:>06d}/{total:>06d}] {100*accuracy:02.3f}% | [{current:>06d}/{no_data_train:>06d}] |")
        accuracy = corrects/total
        tqdm.write(80*'=')
        tqdm.write(
            f'Avg train loss: {train_loss/no_train_batches:03.6f} Accuracy: {100*accuracy:02.3f}%')

        model.eval()

        if create_test_files:
            test_prc = Process(target=test_data_gen)
            test_prc.start()

        test_loss = 0
        corrects = 0
        total = 0
        with torch.no_grad():
            for batch_i in range(no_test_batches):
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
                test_loss += lossfn(pred, y).item() * current_batch_size

                pred_label = torch.argmax(pred, -1)
                corrects += torch.sum(y == pred_label).item()
                total += y.numel()

        accuracy = corrects/total
        test_loss /= no_test_batches
        tqdm.write(80*'=')
        tqdm.write(
            f"Avg Test Error Epoch {epoch_i:04d} Loss: {test_loss:03.6f} Accuracy: {100*accuracy:02.3f}%")

        if min_test_loss > test_loss:
            best_epoch = epoch_i
            min_test_loss = test_loss
            torch.save(model.state_dict(), "model.pth")
            tqdm.write(f'Best Epoch: {best_epoch}! Model Saved')

        if last_test_loss * 0.999 >= test_loss:
            progressive_epoch = epoch_i

        last_test_loss = test_loss

        # Check for early stopping
        # if epoch_i - progressive_epoch >= patience:
        #     tqdm.write(
        #         f'Early stopping after {patience} epochs of no significant improvement')
        #     break
        tqdm.write(80*'=')
        scheduler.step()
