import sys
import numpy as np
from os.path import join, exists
from os import listdir, remove, mkdir
import joblib
from sklearn.utils import gen_batches
from multiprocessing import Process, freeze_support
from model import TransformerEncoder
import torch
from tqdm import tqdm

kmeans_folder = join('..', 'Kmeans of sounds')
sys.path.append(kmeans_folder)
# from slider import slider

path_input = join(kmeans_folder, 'features')
participant = 'sub-06'

melSpec = np.load(join(path_input, f'{participant}_spec.npy'))
feat = np.load(join(path_input, f'{participant}_feat.npy'))
kmeans = joblib.load(join(kmeans_folder, 'kmeans.joblib'))
n_clusters = 50

no_samples = melSpec.shape[0]
w = 100


def train_test_label(no_samples, r=0.2, w=w, p=0.1, seed=15):
    l_test = np.int32(np.round(no_samples*r))
    assert l_test > w
    train_pick_list = list(range(w, no_samples-w-l_test+1))  # + [0, -1]
    np.random.seed(seed)  # reproducibility
    test_indice_clip = np.random.choice(train_pick_list, 1).item()
    train_indices = list(range(0, test_indice_clip-w)) + \
        list(range(test_indice_clip+l_test, no_samples-w+1))
    test_indices = list(range(test_indice_clip, test_indice_clip+l_test-w+1))

    if 0 < p < 1:
        # Only a portion of train data to test the general capability of the model at first, optional
        np.random.seed(28)  # reproducibility
        train_indices = np.random.choice(train_indices, int(len(train_indices)*p), replace=False)
    
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
            x1 = np.transpose(feat[s_ind:s_ind+w])
            y0 = np.expand_dims(melSpec[s_ind:s_ind+w].flatten(), 0)
            y0 = kmeans.predict(y0).item()
            y1 = np.zeros((n_clusters,))
            y1[y0] = 1.
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
            x1 = np.transpose(feat[s_ind:s_ind+w])
            y0 = np.expand_dims(melSpec[s_ind:s_ind+w].flatten(), 0)
            y0 = kmeans.predict(y0).item()
            y1 = np.zeros((n_clusters,))
            y1[y0] = 1.
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
    num_layers = 8
    d_model = 100
    n_head = 10
    d_ff = 100
    dropout = 0.5

    model = TransformerEncoder(num_layers, d_model, n_head, d_ff, n_clusters, dropout).to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    lossfn = torch.nn.CrossEntropyLoss()

    create_test_files = False

    min_test_loss = float('inf')
    last_test_loss = float('inf')
    best_epoch = 0
    progressive_epoch = 0
    patience = 15
    print_batch = int(np.round_(no_train_batches/5))

    prc = Process(target=train_data_gen_proc, args=(epochs,))
    prc.start()

    for epoch_i in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        for batch_i in tqdm(range(no_train_batches)):
            filename = join(train_data_path,
                            f'train_E{epoch_i:04}_B{batch_i:04}.npz')
            while not (exists(filename)):
                pass
            read_flag = False
            while not (read_flag):
                try:
                    z = np.load(filename)
                    read_flag = True
                except:
                    pass
                    # print('Error!')
            y = z['y'].astype('float32')
            X = z['X'].astype('float32')
            current_batch_size = X.shape[0]
            X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)
            z.close()

            # Compute prediction error
            pred = model(X)
            loss = lossfn(pred, y)
            train_loss += loss.item()

            del X, y, z

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            remove(filename)

            if (batch_i + 1) % print_batch == 0:
                loss, current = loss.item(), (batch_i + 1) * batch_size
                tqdm.write(
                    f"loss: {loss/current_batch_size:>7f}  [{current:>04d}/{no_data_train:>04d}]")

        tqdm.write(f'Avg train loss: {train_loss/no_data_train}')

        model.eval()

        if create_test_files:
            test_prc = Process(target=test_data_gen)
            test_prc.start()

        test_loss = 0
        with torch.no_grad():
            for batch_i in range(no_test_batches):
                filename = join(test_data_path, f'test_B{batch_i:04}.npz')
                while not (exists(filename)):
                    pass
                read_flag = False
                while not (read_flag):
                    try:
                        z = np.load(filename)
                        read_flag = True
                    except:
                        pass
                        # print('Error!')
                y = z['y'].astype('float32')
                X = z['X'].astype('float32')
                z.close()
                X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

                pred = model(X)
                test_loss += lossfn(pred, y).item()
                del X, y

        test_loss /= no_data_test
        tqdm.write(
            f"Test Error Epoch {epoch_i:04}: \n Avg loss: {test_loss:>8f} \n")

        if min_test_loss > test_loss:
            best_epoch = epoch_i
            min_test_loss = test_loss
            torch.save(model.state_dict(), "model.pth")
            tqdm.write(f'Best Epoch: {best_epoch}! Model Saved')

        if last_test_loss * 0.99 >= test_loss:
            progressive_epoch = epoch_i

        last_test_loss = test_loss

        # Check for early stopping
        if epoch_i - progressive_epoch >= patience:
            tqdm.write(
                f'Early stopping after {patience} epochs of no significant improvement')
            break
