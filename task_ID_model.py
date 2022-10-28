import time
import argparse
from models import *
from superposition import *
from prepare_data import *
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


class Simple_MLP(nn.Module):

    def __init__(self):
        super(Simple_MLP, self).__init__()

        hidden_size = 32

        self.mlp = nn.Sequential(
            nn.Linear(8192, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 6)
        )


    def forward(self, input):
        input = torch.flatten(input, start_dim=1, end_dim=2)
        output = self.mlp(input)
        return output


def one_hot_6(index):
    x = [0] * 6
    x[index] = 1
    return x



batch_size = 256
num_runs = 1
num_epochs = 10
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for r in range(num_runs):
    print('- - Run %d - -' % (r + 1))

    model = Simple_MLP().to(device)
    print(model)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2,
                                                           threshold=0.0001, min_lr=1e-8, verbose=True)

    Xs = (torch.load('X_0.pt'), torch.load('X_1.pt'), torch.load('X_2.pt'),
          torch.load('X_3.pt'), torch.load('X_4.pt'), torch.load('X_5.pt'))
    ys = [np.repeat([i], len(Xs[i]), axis=0) for i in range(6)]
    # ys = [np.repeat([one_hot_6(i)], len(Xs[i]), axis=0) for i in range(6)]

    X = torch.FloatTensor(np.concatenate(Xs, axis=0)[::3])
    y = torch.FloatTensor(np.concatenate(tuple(ys), axis=0)[::3])

    # split data into train, validation and test set
    # y = torch.max(y, 1)[1]  # change one-hot-encoded vectors to numbers
    permutation = torch.randperm(X.size()[0])
    X = X[permutation]
    y = y[permutation]
    index_val = round(0.8 * len(permutation))
    index_test = round(0.9 * len(permutation))

    X_train, y_train = X[:index_val, :, :], y[:index_val]
    X_val, y_val = X[index_val:index_test, :, :], y[index_val:index_test]
    X_test, y_test = X[index_test:, :, :], y[index_test:]

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    for epoch in range(num_epochs):
        model.train()
        model = model.cuda()

        for batch_X, batch_y in train_loader:
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()
                batch_y = batch_y.cuda()

            outputs = model(batch_X)

            optimizer.zero_grad()
            loss = criterion(outputs, batch_y.long())
            loss.backward()
            optimizer.step()

        # check validation set
        model.eval()
        with torch.no_grad():
            val_outputs = []

            for batch_X, batch_y in val_loader:
                if torch.cuda.is_available():
                    batch_X = batch_X.cuda()

                outputs = model(batch_X)

                val_outputs.append(outputs)

            outputs_all = torch.cat(val_outputs, dim=0)
            true = y_val.cpu().detach().numpy()
            predicted = np.argmax(outputs_all.cpu().detach().numpy(), axis=1).ravel()
            val_acc = np.sum(true == predicted) / true.shape[0]

            # val_acc, val_auroc, val_auprc = get_stats(val_outputs, y_val)
            val_auroc, val_auprc = -1, -1
            val_loss = criterion(torch.cat(val_outputs, dim=0), y_val.long().cuda())

            print("Epoch: %d --- val acc: %.2f, val AUROC: %.2f, val AUPRC: %.2f, val loss: %.3f" %
                  (epoch, val_acc * 100, val_auroc * 100, val_auprc * 100, val_loss))

            scheduler.step(val_auroc)

    # check test set
    model.eval()
    with torch.no_grad():
        test_outputs = []

        for batch_X, batch_y in test_loader:
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()

            outputs = model(batch_X)

            test_outputs.append(outputs)

        outputs_all = torch.cat(test_outputs, dim=0)
        true = y_test.cpu().detach().numpy()
        predicted = np.argmax(outputs_all.cpu().detach().numpy(), axis=1).ravel()
        test_acc = np.sum(true == predicted) / true.shape[0]

        # test_acc, test_auroc, test_auprc = get_stats(test_outputs, y_test)
        # test_auroc, test_auprc = -1, -1

        print("TEST: test acc: %.2f" % (test_acc * 100))

        predicted = np.argmax(torch.cat(test_outputs, dim=0).cpu().detach().numpy(), axis=1).ravel()
        # print('Classification report:', classification_report(y_test.cpu().detach().numpy(), predicted))
        print('Confusion matrix:\n',
              confusion_matrix(y_test.cpu().detach().numpy(), predicted, labels=list(range(6))))






