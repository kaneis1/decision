## Python Imports

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import pearsonr
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import init
import seaborn as sns

sns.set()
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

MSE = mean_squared_error
lag = 1

import matplotlib as mpl
import math

mpl.rcParams.update(mpl.rcParamsDefault)

## Plotting Config

SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 30

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

## IPD Task


def MSE_by_time(r, p):
    err = []
    for t in np.arange(r.shape[1]):
        if len(p.shape) == 3:
            err.append(MSE(r[:, t, :], p[:, t, :]))
        else:
            err.append(MSE(r[:, t, 0], p[:, t]))
    return np.array(err)


def accuracy_by_time(r, p):
    """Calculate accuracy for binary classification at each time step"""
    # Convert torch tensors to numpy arrays if needed
    if torch.is_tensor(r):
        r = r.cpu().numpy()
    if torch.is_tensor(p):
        p = p.cpu().numpy()
    
    # Ensure both are numpy arrays
    r = np.asarray(r)
    p = np.asarray(p)
    
    acc = []
    for t in np.arange(r.shape[1]):
        if len(p.shape) == 3:
            # p is already probabilities, threshold at 0.5
            p_binary = (p[:, t, 0] > 0.5).astype(float)
            # Check if r is 3D before accessing r[:, t, 0]
            r_binary = r[:, t, 0] if len(r.shape) == 3 else r[:, t]
        else:
            # p is already probabilities, threshold at 0.5
            p_binary = (p[:, t] > 0.5).astype(float)
            r_binary = r[:, t, 0] if len(r.shape) == 3 else r[:, t]
        
        # Ensure both are numpy arrays
        p_binary = np.asarray(p_binary)
        r_binary = np.asarray(r_binary)
        
        # Calculate accuracy
        comparison = (p_binary == r_binary)
        acc.append(np.mean(comparison.astype(float)))
    return np.array(acc)


def accuracy_by_time_igt(r, p):
    """Calculate accuracy for multi-class classification (IGT - 4 decks) at each time step"""
    # Convert torch tensors to numpy arrays if needed
    if torch.is_tensor(r):
        r = r.cpu().numpy()
    if torch.is_tensor(p):
        p = p.cpu().numpy()
    
    # Ensure both are numpy arrays
    r = np.asarray(r)
    p = np.asarray(p)
    
    acc = []
    for t in np.arange(r.shape[1]):
        # Get predicted deck (argmax of probabilities)
        p_pred = np.argmax(p[:, t, :], axis=1)
        # Get actual deck (argmax of one-hot encoding)
        r_pred = np.argmax(r[:, t, :], axis=1)
        
        # Calculate accuracy
        comparison = (p_pred == r_pred)
        acc.append(np.mean(comparison.astype(float)))
    return np.array(acc)


def ipd_regression_data(regressiondata):
    x, y = [], []
    for e in regressiondata:
        for t in np.arange(8):
            x.append(
                [
                    e[2],
                    e[3],
                    e[0],
                    e[4],
                    e[1],
                    e[1] * e[2],
                    e[1] * e[3],
                    e[46],
                    e[47],
                    e[46] * e[1],
                    e[14 - t],
                    e[23 - t],
                    e[4] * e[23 - t],
                    t + 2,
                ]
            )
            y.append(np.abs(e[13 - t]))
    return np.array(x), np.array(y)


def valid_ipd(n):
    if n < 1:
        n = int(8258 * n)
    shuffindex = np.random.permutation(8258)
    data = pd.read_csv("./data/IPD/all_data.csv")
    trajs = np.array(data[data["period"] == 10].iloc[:, 9:27])  # (8258, 18)
    regressiondata = np.array(data[data["period"] == 10].iloc[:, 3:51])  # (8258, 48)
    regressiondata, trajs = regressiondata[shuffindex], trajs[shuffindex]
    train_set_rgx, train_set_rgy = ipd_regression_data(regressiondata[n:])
    test_set_rgx, test_set_rgy = ipd_regression_data(regressiondata[:n])
    trajs = trajs.reshape((trajs.shape[0], 2, 9))  # (8258, 2, 9)
    trajs[trajs == 0] = 2
    trajs = trajs - 1
    train_set, test_set = trajs[n:], trajs[:n]
    return train_set, test_set, train_set_rgx, test_set_rgx, train_set_rgy, test_set_rgy


def getCR(r):  # cooperation rate
    if len(r.shape) == 3:
        r = r[:, :, 0]
    cr = np.zeros(r.shape)
    for t in np.arange(r.shape[1]):
        for b in np.arange(r.shape[0]):
            cr[b, t] = r[b, : t + 1].sum() / (t + 1)
    return cr


def ipd_set2arset():
    null_arset = np.zeros((9, 2))
    train_arset = np.zeros((9, 2))
    for ins in train_set:
        ts = ins.copy()
        ts[ts == 0] = -1
        train_arset = np.concatenate((train_arset, null_arset, ts.T), axis=0)
    return train_arset


## BERT Transformer Components

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        pe = self.pe[:seq_len, :, :]
        return x + pe


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(attention_output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class lstmModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layer_num):
        super().__init__()
        self.lstmLayer = nn.LSTM(in_dim, hidden_dim, layer_num)
        self.relu = nn.ReLU()
        self.fcLayer = nn.Linear(hidden_dim, out_dim)
        self.weightInit = np.sqrt(1.0 / hidden_dim)

    def forward(self, x):
        out, _ = self.lstmLayer(x)
        out = self.relu(out)
        out = self.fcLayer(out)
        # Return logits for BCEWithLogitsLoss (no softmax)
        return out


class bertModelIPD(nn.Module):
    """BERT model for IPD binary classification"""
    def __init__(self, in_dim, hidden_dim, num_layers, num_heads=2, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.state_dim = in_dim
        self.input_embedding = nn.Linear(in_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=max_len)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, 1)  # Binary classification: 1 logit
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, in_dim)
        batch_size, seq_len, _ = x.shape
        x = self.input_embedding(x)  # (batch_size, seq_len, hidden_dim)
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        x = self.output_layer(x)  # (batch_size, seq_len, 1)
        return x


n_fold = 5
for fold in np.arange(n_fold):
    (
        train_set,
        test_set,
        train_set_rgx,
        test_set_rgx,
        train_set_rgy,
        test_set_rgy,
    ) = valid_ipd(0.2)
    full_data = {
        "train": train_set,
        "test": test_set,
        "train_set_rgx": train_set_rgx,
        "train_set_rgy": train_set_rgy,
        "test_set_rgx": test_set_rgx,
        "test_set_rgy": test_set_rgy,
    }
    with open("./data/IPD/processed_train_test.pkl", "wb") as handle:
        pickle.dump(full_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    n_nodes, n_layers = 10, 2
    lstm = lstmModel(2, n_nodes, 2, n_layers)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(lstm.parameters(), lr=1e-2)
    n_epochs, window, batch_size = 10, 10, 100
    loss_set = []
    for ep in np.arange(n_epochs):
        for bc in np.arange(train_set.shape[0] / batch_size):
            inputs = Variable(
                torch.from_numpy(
                    train_set[int(bc * batch_size) : int((bc + 1) * batch_size)]
                )
                .transpose(1, 2)
                .float()
            )
            target = Variable(
                torch.from_numpy(
                    train_set[int(bc * batch_size) : int((bc + 1) * batch_size)]
                )
                .transpose(1, 2)
                .float()
            )
            output = lstm(inputs)
            # BCEWithLogitsLoss expects logits and binary targets
            # output shape: (batch, time, 2) -> take first channel for binary classification
            loss = criterion(output.squeeze()[:, :-lag, 0], target[:, lag:, 0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print_loss = loss.item()
            loss_set.append(print_loss)
            if bc % window == 0:
                print(fold)
                print(
                    "Epoch[{}/{}], Batch[{}/{}], Loss: {:.5f}".format(
                        ep + 1,
                        n_epochs,
                        bc + 1,
                        train_set.shape[0] / batch_size,
                        print_loss,
                    )
                )
    lstm = lstm.eval()
    # bert
    bert_ipd = bertModelIPD(2, n_nodes, n_layers, num_heads=2, dropout=0.1)
    criterion_bert = nn.BCEWithLogitsLoss()
    optimizer_bert = optim.Adam(bert_ipd.parameters(), lr=1e-2)
    loss_set_bert = []
    for ep in np.arange(n_epochs):
        for bc in np.arange(train_set.shape[0] / batch_size):
            inputs = Variable(
                torch.from_numpy(
                    train_set[int(bc * batch_size) : int((bc + 1) * batch_size)]
                )
                .transpose(1, 2)
                .float()
            )
            target = Variable(
                torch.from_numpy(
                    train_set[int(bc * batch_size) : int((bc + 1) * batch_size)]
                )
                .transpose(1, 2)
                .float()
            )
            output = bert_ipd(inputs)
            loss = criterion_bert(output.squeeze()[:, :-lag, 0], target[:, lag:, 0])
            optimizer_bert.zero_grad()
            loss.backward()
            optimizer_bert.step()
            print_loss = loss.item()
            loss_set_bert.append(print_loss)
            if bc % window == 0:
                print(fold, "BERT")
                print(
                    "Epoch[{}/{}], Batch[{}/{}], Loss: {:.5f}".format(
                        ep + 1,
                        n_epochs,
                        bc + 1,
                        train_set.shape[0] / batch_size,
                        print_loss,
                    )
                )
    bert_ipd = bert_ipd.eval()
    # ar
    train_arset = ipd_set2arset()
    armodel = VAR(train_arset)
    armodel = armodel.fit()
    px = torch.from_numpy(test_set).transpose(1, 2).float()
    ry = px
    pyar = np.zeros((px.shape[0], px.shape[1]))
    for i in np.arange(px.shape[0]):
        for t in np.arange(px.shape[1]):
            pyar[i, t] = (
                1 if armodel.forecast(np.array(px[i, : t + 1]), lag)[0][0] > 0 else 0
            )
    # lr
    lrmodel = LogisticRegression(random_state=0, max_iter=1000).fit(
        train_set_rgx, train_set_rgy
    )
    pylrraw = lrmodel.predict(test_set_rgx)
    pylr = pylrraw.reshape((1651, 8))
    varX = Variable(px)
    py_logits = lstm(varX).squeeze().data.cpu().numpy()
    # Convert logits to probabilities for evaluation
    py = torch.sigmoid(torch.from_numpy(py_logits)).numpy()
    # bert predictions
    pybert_logits = bert_ipd(varX).squeeze().data.cpu().numpy()
    pybert = torch.sigmoid(torch.from_numpy(pybert_logits)).numpy()
    if fold == 0:
        test_set_full = test_set
        py_full = py
        pyar_full = pyar
        pylr_full = pylr
        pybert_full = pybert
    else:
        test_set_full = np.concatenate((test_set_full, test_set))
        px = torch.from_numpy(test_set_full).transpose(1, 2).float()
        ry = px
        py_full = np.concatenate((py_full, py))
        pyar_full = np.concatenate((pyar_full, pyar))
        pylr_full = np.concatenate((pylr_full, pylr))
        pybert_full = np.concatenate((pybert_full, pybert))
        py = py_full
        pyar = pyar_full
        pylr = pylr_full
        pybert = pybert_full

ryc = getCR(ry[:, lag:])
pyc = getCR(py[:, :-lag])
pycar = getCR(pyar[:, :-lag])
pyclr = getCR(pylr)
pycbert = getCR(pybert[:, :-lag, 0])

plt.clf()
plt.plot(np.arange(8) + 1, accuracy_by_time(ry[:, lag:], py[:, :-lag, 0]), "r", label="LSTM")
plt.plot(np.arange(8) + 1, accuracy_by_time(ry[:, lag:], pyar[:, :-lag]), "b", label="AR")
plt.plot(np.arange(8) + 1, accuracy_by_time(ry[:, lag:], pylr), "g", label="LR")
plt.plot(np.arange(8) + 1, accuracy_by_time(ry[:, lag:], pybert[:, :-lag, 0]), "m", label="BERT")
plt.legend(loc="best")
plt.title("IPD Task - Action Prediction Accuracy")
plt.xlabel("Prediction Time Steps")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(
    "Figures/ipd_accuracy_nodes_" + str(n_nodes) + "_layers_" + str(n_layers) + ".png"
)

lstm_acc = np.mean(accuracy_by_time(ry[:, lag:], py[:, :-lag, 0]))
ar_acc = np.mean(accuracy_by_time(ry[:, lag:], pyar[:, :-lag]))
lr_acc = np.mean(accuracy_by_time(ry[:, lag:], pylr))
bert_acc = np.mean(accuracy_by_time(ry[:, lag:], pybert[:, :-lag, 0]))
print("IPD Accuracy - LSTM:", lstm_acc, "AR:", ar_acc, "LR:", lr_acc, "BERT:", bert_acc)

plt.clf()
plt.plot(np.arange(pyc.shape[1]) + 1, np.array(pyc.mean(0)), "r", label="LSTM")
plt.plot(np.arange(pycar.shape[1]) + 1, np.array(pycar.mean(0)), "b", label="AR")
plt.plot(np.arange(pyclr.shape[1]) + 1, np.array(pyclr.mean(0)), "g", label="LR")
plt.plot(np.arange(pycbert.shape[1]) + 1, np.array(pycbert.mean(0)), "m", label="BERT")
plt.plot(np.arange(ryc.shape[1]) + 1, np.array(ryc.mean(0)), "k", label="Human")
plt.fill_between(
    np.arange(pyc.shape[1]) + 1,
    np.array(pyc.mean(0)) - np.array(pyc.std(0)) / np.sqrt(pyc.shape[0] / n_fold),
    np.array(pyc.mean(0)) + np.array(pyc.std(0)) / np.sqrt(pyc.shape[0] / n_fold),
    color="r",
    alpha=0.2,
)
plt.fill_between(
    np.arange(pycar.shape[1]) + 1,
    np.array(pycar.mean(0)) - np.array(pycar.std(0)) / np.sqrt(pycar.shape[0] / n_fold),
    np.array(pycar.mean(0)) + np.array(pycar.std(0)) / np.sqrt(pycar.shape[0] / n_fold),
    color="b",
    alpha=0.2,
)
plt.fill_between(
    np.arange(pyclr.shape[1]) + 1,
    np.array(pyclr.mean(0)) - np.array(pyclr.std(0)) / np.sqrt(pyclr.shape[0] / n_fold),
    np.array(pyclr.mean(0)) + np.array(pyclr.std(0)) / np.sqrt(pyclr.shape[0] / n_fold),
    color="g",
    alpha=0.2,
)
plt.fill_between(
    np.arange(pycbert.shape[1]) + 1,
    np.array(pycbert.mean(0)) - np.array(pycbert.std(0)) / np.sqrt(pycbert.shape[0] / n_fold),
    np.array(pycbert.mean(0)) + np.array(pycbert.std(0)) / np.sqrt(pycbert.shape[0] / n_fold),
    color="m",
    alpha=0.2,
)
plt.fill_between(
    np.arange(ryc.shape[1]) + 1,
    np.array(ryc.mean(0)) - np.array(ryc.std(0)) / np.sqrt(ryc.shape[0] / n_fold),
    np.array(ryc.mean(0)) + np.array(ryc.std(0)) / np.sqrt(ryc.shape[0] / n_fold),
    color="k",
    alpha=0.2,
)
plt.legend(loc="best")
plt.ylim([0, 1])
plt.title("IPD Task - Cooperation Prediction")
plt.xlabel("Prediction Time Steps")
plt.ylabel("Cooperation Rates")
plt.savefig(
    "Figures/ipd_coop_nodes_" + str(n_nodes) + "_layers_" + str(n_layers) + ".png"
)

plt.clf()
plt.plot(loss_set)
plt.title("IPD Task - LSTM loss")
plt.xlabel("batch")
plt.ylabel("loss")
plt.savefig(
    "Figures/ipd_lstm_loss_nodes_" + str(n_nodes) + "_layers_" + str(n_layers) + ".png"
)
# plt.show()

plt.clf()
plt.plot(np.arange(pyc.shape[1]) + 1, np.array(pyc.mean(0)), "r", label="prediction")
plt.plot(np.arange(ryc.shape[1]) + 1, np.array(ryc.mean(0)), "b", label="real")
plt.fill_between(
    np.arange(pyc.shape[1]) + 1,
    np.array(pyc.mean(0)) - np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    np.array(pyc.mean(0)) + np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    color="r",
    alpha=0.2,
)
plt.fill_between(
    np.arange(ryc.shape[1]) + 1,
    np.array(ryc.mean(0)) - np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    np.array(ryc.mean(0)) + np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    color="b",
    alpha=0.2,
)
plt.legend(loc="best")
plt.ylim([0, 1])
plt.title("IPD Task - Predict Cooperation by LSTM")
plt.xlabel("Prediction Time Steps")
plt.ylabel("Cooperation Rates")
plt.savefig(
    "Figures/ipd_lstm_coop_nodes_" + str(n_nodes) + "_layers_" + str(n_layers) + ".png"
)
# plt.show()

plt.clf()
plt.plot(np.arange(ryc.shape[1]) + 1, np.array(pycar.mean(0)), "r", label="prediction")
plt.plot(np.arange(ryc.shape[1]) + 1, np.array(ryc.mean(0)), "b", label="real")
plt.fill_between(
    np.arange(pycar.shape[1]) + 1,
    np.array(pycar.mean(0)) - np.array(pycar.std(0)) / np.sqrt(pycar.shape[0]),
    np.array(pycar.mean(0)) + np.array(pycar.std(0)) / np.sqrt(pycar.shape[0]),
    color="r",
    alpha=0.2,
)
plt.fill_between(
    np.arange(ryc.shape[1]) + 1,
    np.array(ryc.mean(0)) - np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    np.array(ryc.mean(0)) + np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    color="b",
    alpha=0.2,
)
plt.legend(loc="best")
plt.ylim([0, 1])
plt.title("IPD Task - Predict Cooperation by AR")
plt.xlabel("Prediction Time Steps")
plt.ylabel("Cooperation Rates")
plt.savefig(
    "Figures/ipd_ar_coop_nodes_" + str(n_nodes) + "_layers_" + str(n_layers) + ".png"
)

plt.clf()
plt.plot(np.arange(ryc.shape[1]) + 1, np.array(pyclr.mean(0)), "r", label="prediction")
plt.plot(np.arange(ryc.shape[1]) + 1, np.array(ryc.mean(0)), "b", label="real")
plt.fill_between(
    np.arange(pyclr.shape[1]) + 1,
    np.array(pyclr.mean(0)) - np.array(pyclr.std(0)) / np.sqrt(pyclr.shape[0]),
    np.array(pyclr.mean(0)) + np.array(pyclr.std(0)) / np.sqrt(pyclr.shape[0]),
    color="r",
    alpha=0.2,
)
plt.fill_between(
    np.arange(ryc.shape[1]) + 1,
    np.array(ryc.mean(0)) - np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    np.array(ryc.mean(0)) + np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    color="b",
    alpha=0.2,
)
plt.legend(loc="best")
plt.ylim([0, 1])
plt.title("IPD Task - Predict Cooperation by LR")
plt.xlabel("Prediction Time Steps")
plt.ylabel("Cooperation Rates")
plt.savefig(
    "Figures/ipd_lr_coop_nodes_" + str(n_nodes) + "_layers_" + str(n_layers) + ".png"
)

## IGT Task

choice_95 = pd.read_csv("./data/IGT/choice_95.csv", delimiter=",")
choice_100 = pd.read_csv("./data/IGT/choice_100.csv", delimiter=",")
choice_150 = pd.read_csv("./data/IGT/choice_150.csv", delimiter=",")


def getTS(r):
    ts = np.zeros((r.shape[0], r.shape[1], 4))
    for i in np.arange(4):
        ts[r == i + 1, i] = 1
    return ts


def revTS(r):
    ts = r
    for t in np.arange(ts.shape[1]):
        for b in np.arange(ts.shape[0]):
            choice = np.argmax(ts[b, t, :])
            ts[b, t, :] = 0
            ts[b, t, choice] = 1
    for i in np.arange(4):
        ts[:, :, i] = ts[:, :, i] * (i + 1)
    ts = ts.sum(2)
    return ts


def getChR(r):  # choice rate
    ts = r
    for t in np.arange(ts.shape[1]):
        for b in np.arange(ts.shape[0]):
            choice = np.argmax(ts[b, t, :])
            ts[b, t, :] = 0
            ts[b, t, choice] = 1
    cr = np.zeros(r.shape)
    for i in np.arange(ts.shape[2]):
        for t in np.arange(ts.shape[1]):
            for b in np.arange(ts.shape[0]):
                cr[b, t, i] = ts[b, : t + 1, i].sum() / (t + 1)
    return cr


def getCoR(r):  # correct rate
    ts = r
    for t in np.arange(ts.shape[1]):
        for b in np.arange(ts.shape[0]):
            choice = np.argmax(ts[b, t, :])
            ts[b, t, :] = 0
            ts[b, t, choice] = 1
    ts[:, :, 2] = ts[:, :, 2] + ts[:, :, 3]
    cr = np.zeros((r.shape[0], r.shape[1]))
    for t in np.arange(ts.shape[1]):
        for b in np.arange(ts.shape[0]):
            cr[b, t] = ts[b, : t + 1, 2].sum() / (t + 1)
    return cr


def valid_igt(n):
    if n < 1:
        n = int(617 * n)
    set_100 = np.array(getTS(choice_100))[:, :94, :]
    set_95 = np.array(getTS(choice_95))[:, :94, :]
    set_150 = np.array(getTS(choice_150))[:, :94, :]
    full_set = np.concatenate((set_100, set_95, set_150), axis=0)
    np.random.shuffle(full_set)
    return full_set[:n], full_set[n:]


def igt_set2arset():
    null_arset = -np.ones((94, 4))
    train_arset_igt = -np.ones((94, 4))
    for ins in train_set_igt:
        train_arset_igt = np.concatenate((train_arset_igt, null_arset, ins), axis=0)
    return train_arset_igt


class lstmIGT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layer_num):
        super().__init__()
        self.lstmLayer = nn.LSTM(in_dim, hidden_dim, layer_num)
        self.relu = nn.ReLU()
        self.fcLayer = nn.Linear(hidden_dim, out_dim)
        self.weightInit = np.sqrt(1.0 / hidden_dim)

    def forward(self, x):
        out, _ = self.lstmLayer(x)
        out = self.relu(out)
        out = self.fcLayer(out)
        out = nn.Softmax(dim=-1)(out)
        return out


class bertModelIGT(nn.Module):
    """BERT model for IGT multi-class classification"""
    def __init__(self, in_dim, hidden_dim, num_layers, out_dim, num_heads=2, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.state_dim = in_dim
        self.input_embedding = nn.Linear(in_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=max_len)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, out_dim)  # Multi-class: out_dim logits
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, in_dim)
        batch_size, seq_len, _ = x.shape
        x = self.input_embedding(x)  # (batch_size, seq_len, hidden_dim)
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        x = self.output_layer(x)  # (batch_size, seq_len, out_dim)
        x = nn.Softmax(dim=-1)(x)  # Apply softmax for multi-class
        return x


n_fold = 5
for fold in np.arange(n_fold):
    test_set_igt, train_set_igt = valid_igt(0.2)
    # lstm
    n_nodes, n_layers = 10, 2
    lstm_igt = lstmIGT(4, n_nodes, 4, n_layers)
    criterion_igt = nn.MSELoss()
    optimizer_igt = optim.Adam(lstm_igt.parameters(), lr=1e-2)
    n_epochs, window, batch_size = 100, 10, 10
    loss_set_igt = []
    for ep in np.arange(n_epochs):
        for bc in np.arange(train_set_igt.shape[0] / batch_size):
            inputs = Variable(
                torch.from_numpy(
                    train_set_igt[int(bc * batch_size) : int((bc + 1) * batch_size)]
                ).float()
            )
            target = Variable(
                torch.from_numpy(
                    train_set_igt[int(bc * batch_size) : int((bc + 1) * batch_size)]
                ).float()
            )
            output = lstm_igt(inputs)
            loss = criterion_igt(output[:, :-lag], target[:, lag:])
            optimizer_igt.zero_grad()
            loss.backward()
            optimizer_igt.step()
            print_loss = loss.item()
            loss_set_igt.append(print_loss)
            if bc % window == 0:
                print(fold)
                print(
                    "Epoch[{}/{}], Batch[{}/{}], Loss: {:.5f}".format(
                        ep + 1,
                        n_epochs,
                        bc + 1,
                        train_set_igt.shape[0] / batch_size,
                        print_loss,
                    )
                )
    lstm_igt = lstm_igt.eval()
    # bert
    bert_igt = bertModelIGT(4, n_nodes, n_layers, 4, num_heads=2, dropout=0.1)
    criterion_bert_igt = nn.MSELoss()
    optimizer_bert_igt = optim.Adam(bert_igt.parameters(), lr=1e-2)
    loss_set_bert_igt = []
    for ep in np.arange(n_epochs):
        for bc in np.arange(train_set_igt.shape[0] / batch_size):
            inputs = Variable(
                torch.from_numpy(
                    train_set_igt[int(bc * batch_size) : int((bc + 1) * batch_size)]
                ).float()
            )
            target = Variable(
                torch.from_numpy(
                    train_set_igt[int(bc * batch_size) : int((bc + 1) * batch_size)]
                ).float()
            )
            output = bert_igt(inputs)
            loss = criterion_bert_igt(output[:, :-lag], target[:, lag:])
            optimizer_bert_igt.zero_grad()
            loss.backward()
            optimizer_bert_igt.step()
            print_loss = loss.item()
            loss_set_bert_igt.append(print_loss)
            if bc % window == 0:
                print(fold, "BERT IGT")
                print(
                    "Epoch[{}/{}], Batch[{}/{}], Loss: {:.5f}".format(
                        ep + 1,
                        n_epochs,
                        bc + 1,
                        train_set_igt.shape[0] / batch_size,
                        print_loss,
                    )
                )
    bert_igt = bert_igt.eval()
    # ar
    train_arset_igt = igt_set2arset()
    armodel_igt = VAR(train_arset_igt)
    armodel_igt = armodel_igt.fit()
    # eval
    px2 = torch.from_numpy(test_set_igt).float()
    ry2 = torch.from_numpy(test_set_igt).float()
    pyar2 = np.zeros(px2.shape)
    for i in np.arange(px2.shape[0]):
        for t in np.arange(px2.shape[1]):
            pyar2[i, t, :] = armodel_igt.forecast(np.array(px2[i, : t + 1]), lag)
    varX = Variable(px2)
    py2 = lstm_igt(varX).data.cpu().numpy()
    py2bert = bert_igt(varX).data.cpu().numpy()
    if fold == 0:
        test_set_igt_full = test_set_igt
        py2_full = py2
        pyar2_full = pyar2
        py2bert_full = py2bert
    else:
        test_set_igt_full = np.concatenate((test_set_igt_full, test_set_igt))
        px2 = torch.from_numpy(test_set_igt_full).float()
        ry2 = torch.from_numpy(test_set_igt_full).float()
        py2_full = np.concatenate((py2_full, py2))
        pyar2_full = np.concatenate((pyar2_full, pyar2))
        py2bert_full = np.concatenate((py2bert_full, py2bert))
        py2 = py2_full
        pyar2 = pyar2_full
        py2bert = py2bert_full

ry2_np = ry2[:, lag:, :].cpu().numpy().copy()
ryc2 = revTS(ry2[:, lag:].cpu().numpy().copy())
ryr2 = getChR(ry2_np)
ryo2 = getCoR(ry2_np)
pyc2 = revTS(py2[:, :-lag].copy())
pyr2 = getChR(py2[:, :-lag, :].copy())
pyo2 = getCoR(py2[:, :-lag, :].copy())
pycar2 = revTS(pyar2[:, :-lag].copy())
pyrar2 = getChR(pyar2[:, :-lag, :].copy())
pyoar2 = getCoR(pyar2[:, :-lag, :].copy())
pyo2bert = getCoR(py2bert[:, :-lag, :].copy())

pyr2bert = getChR(py2bert[:, :-lag, :].copy())

plt.clf()
fig = plt.figure(figsize=(6.4, 4.8))
plt.plot(np.arange(ryr2.shape[1]) + 1, MSE_by_time(ryr2, pyr2), "r", label="LSTM")
plt.plot(np.arange(ryr2.shape[1]) + 1, MSE_by_time(ryr2, pyrar2), "b", label="AR")
plt.plot(np.arange(ryr2.shape[1]) + 1, MSE_by_time(ryr2, pyr2bert), "m", label="BERT")
plt.legend(loc="best")
plt.title("IGT Task - Action Prediction MSE")
plt.xlabel("Prediction Time Steps")
plt.ylabel("MSE")
plt.tight_layout()
plt.savefig(
    "Figures/igt_mse_nodes_" + str(n_nodes) + "_layers_" + str(n_layers) + ".png"
)

igt_lstm_mse = np.mean(MSE_by_time(ryr2, pyr2))
igt_ar_mse = np.mean(MSE_by_time(ryr2, pyrar2))
igt_bert_mse = np.mean(MSE_by_time(ryr2, pyr2bert))
print(n_nodes, n_layers, "MSE - LSTM:", igt_lstm_mse, "AR:", igt_ar_mse, "BERT:", igt_bert_mse)
# 10 2 MSE: 0.014868835952435816 0.02016487523918092

plt.clf()
fig = plt.figure(figsize=(6.4, 4.8))
plt.plot(np.arange(ryr2.shape[1]) + 1, accuracy_by_time_igt(ry2_np, py2[:, :-lag, :]), "r", label="LSTM")
plt.plot(np.arange(ryr2.shape[1]) + 1, accuracy_by_time_igt(ry2_np, pyar2[:, :-lag, :]), "b", label="AR")
plt.plot(np.arange(ryr2.shape[1]) + 1, accuracy_by_time_igt(ry2_np, py2bert[:, :-lag, :]), "m", label="BERT")
plt.legend(loc="best")
plt.title("IGT Task - Action Prediction Accuracy")
plt.xlabel("Prediction Time Steps")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(
    "Figures/igt_accuracy_nodes_" + str(n_nodes) + "_layers_" + str(n_layers) + ".png"
)

igt_lstm_acc = np.mean(accuracy_by_time_igt(ry2_np, py2[:, :-lag, :]))
igt_ar_acc = np.mean(accuracy_by_time_igt(ry2_np, pyar2[:, :-lag, :]))
igt_bert_acc = np.mean(accuracy_by_time_igt(ry2_np, py2bert[:, :-lag, :]))
print(n_nodes, n_layers, "Accuracy - LSTM:", igt_lstm_acc, "AR:", igt_ar_acc, "BERT:", igt_bert_acc)

plt.clf()
plt.plot(np.arange(pyo2.shape[1]) + 1, np.array(pyo2.mean(0)), "r", label="LSTM")
plt.plot(np.arange(ryo2.shape[1]) + 1, np.array(pyoar2.mean(0)), "b", label="AR")
plt.plot(np.arange(ryo2.shape[1]) + 1, np.array(pyo2bert.mean(0)), "m", label="BERT")
plt.plot(np.arange(ryo2.shape[1]) + 1, np.array(ryo2.mean(0)), "k", label="Human")
plt.fill_between(
    np.arange(pyo2.shape[1]) + 1,
    np.array(pyo2.mean(0)) - np.array(pyo2.std(0)) / np.sqrt(pyo2.shape[0] / n_fold),
    np.array(pyo2.mean(0)) + np.array(pyo2.std(0)) / np.sqrt(pyo2.shape[0] / n_fold),
    color="r",
    alpha=0.2,
)
plt.fill_between(
    np.arange(pyoar2.shape[1]) + 1,
    np.array(pyoar2.mean(0))
    - np.array(pyoar2.std(0)) / np.sqrt(pyoar2.shape[0] / n_fold),
    np.array(pyoar2.mean(0))
    + np.array(pyoar2.std(0)) / np.sqrt(pyoar2.shape[0] / n_fold),
    color="b",
    alpha=0.2,
)
plt.fill_between(
    np.arange(pyo2bert.shape[1]) + 1,
    np.array(pyo2bert.mean(0))
    - np.array(pyo2bert.std(0)) / np.sqrt(pyo2bert.shape[0] / n_fold),
    np.array(pyo2bert.mean(0))
    + np.array(pyo2bert.std(0)) / np.sqrt(pyo2bert.shape[0] / n_fold),
    color="m",
    alpha=0.2,
)
plt.fill_between(
    np.arange(ryo2.shape[1]) + 1,
    np.array(ryo2.mean(0)) - np.array(ryo2.std(0)) / np.sqrt(ryo2.shape[0] / n_fold),
    np.array(ryo2.mean(0)) + np.array(ryo2.std(0)) / np.sqrt(ryo2.shape[0] / n_fold),
    color="k",
    alpha=0.2,
)
plt.legend(loc="best")
plt.ylim([0, 1])
plt.title("IGT Task - Choosing Better Decks by AR")
plt.xlabel("Prediction Time Steps")
plt.ylabel("Percentage of Better Decks")
plt.tight_layout()
plt.savefig(
    "Figures/igt_corr_nodes_" + str(n_nodes) + "_layers_" + str(n_layers) + ".png"
)

plt.clf()
fig = plt.figure(figsize=(15, 10))
decks = ["A", "B", "C", "D"]
for i, d in enumerate(decks):
    ax = fig.add_subplot(2, 2, i + 1)
    pyca, pycb, pycbert, ryc = pyr2[:, :, i], pyrar2[:, :, i], pyr2bert[:, :, i], ryr2[:, :, i]
    ax.plot(np.arange(ryc.shape[1]) + 1, np.array(pyca.mean(0)), "r", label="LSTM")
    ax.plot(np.arange(ryc.shape[1]) + 1, np.array(pycb.mean(0)), "b", label="AR")
    ax.plot(np.arange(ryc.shape[1]) + 1, np.array(pycbert.mean(0)), "m", label="BERT")
    ax.plot(np.arange(ryc.shape[1]) + 1, np.array(ryc.mean(0)), "k", label="Human")
    ax.fill_between(
        np.arange(ryc.shape[1]) + 1,
        np.array(pyca.mean(0)) - np.array(pyca.std(0)) / np.sqrt(pyca.shape[0]),
        np.array(pyca.mean(0)) + np.array(pyca.std(0)) / np.sqrt(pyca.shape[0]),
        color="r",
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(ryc.shape[1]) + 1,
        np.array(pycb.mean(0)) - np.array(pycb.std(0)) / np.sqrt(pycb.shape[0]),
        np.array(pycb.mean(0)) + np.array(pycb.std(0)) / np.sqrt(pycb.shape[0]),
        color="b",
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(ryc.shape[1]) + 1,
        np.array(pycbert.mean(0)) - np.array(pycbert.std(0)) / np.sqrt(pycbert.shape[0]),
        np.array(pycbert.mean(0)) + np.array(pycbert.std(0)) / np.sqrt(pycbert.shape[0]),
        color="m",
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(ryc.shape[1]) + 1,
        np.array(ryc.mean(0)) - np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
        np.array(ryc.mean(0)) + np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
        color="k",
        alpha=0.2,
    )
    ax.legend(loc="best")
    ax.set_title("Deck " + d)
    ax.set_xlabel("Prediction Time Steps")
    ax.set_ylabel("Choice Rates")
fig.suptitle("IGT Task - Action Prediction")
fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.savefig(
    "Figures/igt_pred_nodes_" + str(n_nodes) + "_layers_" + str(n_layers) + ".png"
)

plt.clf()
plt.plot(loss_set_igt)
plt.title("IGT Task - LSTM loss")
plt.xlabel("batch")
plt.ylabel("loss")
plt.savefig(
    "Figures/igt_lstm_loss_nodes_" + str(n_nodes) + "_layers_" + str(n_layers) + ".png"
)
# plt.show()

plt.clf()
ryc, pyc = ryo2, pyoar2
plt.plot(np.arange(ryc.shape[1]) + 1, np.array(pyc.mean(0)), "r", label="prediction")
plt.plot(np.arange(ryc.shape[1]) + 1, np.array(ryc.mean(0)), "b", label="real")
plt.fill_between(
    np.arange(pyc.shape[1]) + 1,
    np.array(pyc.mean(0)) - np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    np.array(pyc.mean(0)) + np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    color="r",
    alpha=0.2,
)
plt.fill_between(
    np.arange(ryc.shape[1]) + 1,
    np.array(ryc.mean(0)) - np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    np.array(ryc.mean(0)) + np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    color="b",
    alpha=0.2,
)
plt.legend(loc="best")
plt.ylim([0, 1])
plt.title("IGT Task - Choosing Better Decks by AR")
plt.xlabel("Prediction Time Steps")
plt.ylabel("Percentage of Better Decks")
plt.tight_layout()
plt.savefig(
    "Figures/igt_ar_corr_nodes_" + str(n_nodes) + "_layers_" + str(n_layers) + ".png"
)

plt.clf()
ryc, pyc = ryo2, pyo2
plt.plot(np.arange(ryc.shape[1]) + 1, np.array(pyc.mean(0)), "r", label="prediction")
plt.plot(np.arange(ryc.shape[1]) + 1, np.array(ryc.mean(0)), "b", label="real")
plt.fill_between(
    np.arange(pyc.shape[1]) + 1,
    np.array(pyc.mean(0)) - np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    np.array(pyc.mean(0)) + np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    color="r",
    alpha=0.2,
)
plt.fill_between(
    np.arange(ryc.shape[1]) + 1,
    np.array(ryc.mean(0)) - np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    np.array(ryc.mean(0)) + np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    color="b",
    alpha=0.2,
)
plt.legend(loc="best")
plt.ylim([0, 1])
plt.title("IGT Task - Choosing Better Decks by LSTM")
plt.xlabel("Prediction Time Steps")
plt.ylabel("Percentage of Better Decks")
plt.tight_layout()
plt.savefig(
    "Figures/igt_lstm_corr_nodes_" + str(n_nodes) + "_layers_" + str(n_layers) + ".png"
)

plt.clf()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(221)
pyc, ryc = pyrar2[:, :, 0], ryr2[:, :, 0]
ax.plot(np.arange(ryc.shape[1]) + 1, np.array(pyc.mean(0)), "r", label="prediction")
ax.plot(np.arange(ryc.shape[1]) + 1, np.array(ryc.mean(0)), "b", label="real")
ax.fill_between(
    np.arange(pyc.shape[1]) + 1,
    np.array(pyc.mean(0)) - np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    np.array(pyc.mean(0)) + np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    color="r",
    alpha=0.2,
)
ax.fill_between(
    np.arange(ryc.shape[1]) + 1,
    np.array(ryc.mean(0)) - np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    np.array(ryc.mean(0)) + np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    color="b",
    alpha=0.2,
)
ax.legend(loc="best")
ax.set_title("Deck A")
ax.set_xlabel("Prediction Time Steps")
ax.set_ylabel("Choice Rates")

ax = fig.add_subplot(222)
pyc, ryc = pyrar2[:, :, 1], ryr2[:, :, 1]
ax.plot(np.arange(ryc.shape[1]) + 1, np.array(pyc.mean(0)), "r", label="prediction")
ax.plot(np.arange(ryc.shape[1]) + 1, np.array(ryc.mean(0)), "b", label="real")
ax.fill_between(
    np.arange(pyc.shape[1]) + 1,
    np.array(pyc.mean(0)) - np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    np.array(pyc.mean(0)) + np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    color="r",
    alpha=0.2,
)
ax.fill_between(
    np.arange(ryc.shape[1]) + 1,
    np.array(ryc.mean(0)) - np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    np.array(ryc.mean(0)) + np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    color="b",
    alpha=0.2,
)
ax.legend(loc="best")
ax.set_title("Deck B")
ax.set_xlabel("Prediction Time Steps")
ax.set_ylabel("Choice Rates")

ax = fig.add_subplot(223)
pyc, ryc = pyrar2[:, :, 2], ryr2[:, :, 2]
ax.plot(np.arange(ryc.shape[1]) + 1, np.array(pyc.mean(0)), "r", label="prediction")
ax.plot(np.arange(ryc.shape[1]) + 1, np.array(ryc.mean(0)), "b", label="real")
ax.fill_between(
    np.arange(pyc.shape[1]) + 1,
    np.array(pyc.mean(0)) - np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    np.array(pyc.mean(0)) + np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    color="r",
    alpha=0.2,
)
ax.fill_between(
    np.arange(ryc.shape[1]) + 1,
    np.array(ryc.mean(0)) - np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    np.array(ryc.mean(0)) + np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    color="b",
    alpha=0.2,
)
ax.legend(loc="best")
ax.set_title("Deck C")
ax.set_xlabel("Prediction Time Steps")
ax.set_ylabel("Choice Rates")

ax = fig.add_subplot(224)
pyc, ryc = pyrar2[:, :, 3], ryr2[:, :, 3]
ax.plot(np.arange(ryc.shape[1]) + 1, np.array(pyc.mean(0)), "r", label="prediction")
ax.plot(np.arange(ryc.shape[1]) + 1, np.array(ryc.mean(0)), "b", label="real")
ax.fill_between(
    np.arange(pyc.shape[1]) + 1,
    np.array(pyc.mean(0)) - np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    np.array(pyc.mean(0)) + np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    color="r",
    alpha=0.2,
)
ax.fill_between(
    np.arange(ryc.shape[1]) + 1,
    np.array(ryc.mean(0)) - np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    np.array(ryc.mean(0)) + np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    color="b",
    alpha=0.2,
)
ax.legend(loc="best")
ax.set_title("Deck D")
ax.set_xlabel("Prediction Time Steps")
ax.set_ylabel("Choice Rates")

fig.suptitle("IGT Task - Action Prediction by AR")
fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.savefig(
    "Figures/igt_ar_pred_nodes_" + str(n_nodes) + "_layers_" + str(n_layers) + ".png"
)

plt.clf()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(221)
pyc, ryc = pyr2[:, :, 0], ryr2[:, :, 0]
ax.plot(np.arange(ryc.shape[1]) + 1, np.array(pyc.mean(0)), "r", label="prediction")
ax.plot(np.arange(ryc.shape[1]) + 1, np.array(ryc.mean(0)), "b", label="real")
ax.fill_between(
    np.arange(pyc.shape[1]) + 1,
    np.array(pyc.mean(0)) - np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    np.array(pyc.mean(0)) + np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    color="r",
    alpha=0.2,
)
ax.fill_between(
    np.arange(ryc.shape[1]) + 1,
    np.array(ryc.mean(0)) - np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    np.array(ryc.mean(0)) + np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    color="b",
    alpha=0.2,
)
ax.legend(loc="best")
ax.set_title("Deck A")
ax.set_xlabel("Prediction Time Steps")
ax.set_ylabel("Choice Rates")

ax = fig.add_subplot(222)
pyc, ryc = pyr2[:, :, 1], ryr2[:, :, 1]
ax.plot(np.arange(ryc.shape[1]) + 1, np.array(pyc.mean(0)), "r", label="prediction")
ax.plot(np.arange(ryc.shape[1]) + 1, np.array(ryc.mean(0)), "b", label="real")
ax.fill_between(
    np.arange(pyc.shape[1]) + 1,
    np.array(pyc.mean(0)) - np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    np.array(pyc.mean(0)) + np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    color="r",
    alpha=0.2,
)
ax.fill_between(
    np.arange(ryc.shape[1]) + 1,
    np.array(ryc.mean(0)) - np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    np.array(ryc.mean(0)) + np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    color="b",
    alpha=0.2,
)
ax.legend(loc="best")
ax.set_title("Deck B")
ax.set_xlabel("Prediction Time Steps")
ax.set_ylabel("Choice Rates")

ax = fig.add_subplot(223)
pyc, ryc = pyr2[:, :, 2], ryr2[:, :, 2]
ax.plot(np.arange(ryc.shape[1]) + 1, np.array(pyc.mean(0)), "r", label="prediction")
ax.plot(np.arange(ryc.shape[1]) + 1, np.array(ryc.mean(0)), "b", label="real")
ax.fill_between(
    np.arange(pyc.shape[1]) + 1,
    np.array(pyc.mean(0)) - np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    np.array(pyc.mean(0)) + np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    color="r",
    alpha=0.2,
)
ax.fill_between(
    np.arange(ryc.shape[1]) + 1,
    np.array(ryc.mean(0)) - np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    np.array(ryc.mean(0)) + np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    color="b",
    alpha=0.2,
)
ax.legend(loc="best")
ax.set_title("Deck C")
ax.set_xlabel("Prediction Time Steps")
ax.set_ylabel("Choice Rates")

ax = fig.add_subplot(224)
pyc, ryc = pyr2[:, :, 3], ryr2[:, :, 3]
ax.plot(np.arange(ryc.shape[1]) + 1, np.array(pyc.mean(0)), "r", label="prediction")
ax.plot(np.arange(ryc.shape[1]) + 1, np.array(ryc.mean(0)), "b", label="real")
ax.fill_between(
    np.arange(pyc.shape[1]) + 1,
    np.array(pyc.mean(0)) - np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    np.array(pyc.mean(0)) + np.array(pyc.std(0)) / np.sqrt(pyc.shape[0]),
    color="r",
    alpha=0.2,
)
ax.fill_between(
    np.arange(ryc.shape[1]) + 1,
    np.array(ryc.mean(0)) - np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    np.array(ryc.mean(0)) + np.array(ryc.std(0)) / np.sqrt(ryc.shape[0]),
    color="b",
    alpha=0.2,
)
ax.legend(loc="best")
ax.set_title("Deck D")
ax.set_xlabel("Prediction Time Steps")
ax.set_ylabel("Choice Rates")

fig.suptitle("IGT Task - Action Prediction by LSTM")
fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.savefig(
    "Figures/igt_lstm_pred_nodes_" + str(n_nodes) + "_layers_" + str(n_layers) + ".png"
)

# autocorrelation

autocorrelation_plot(ry.transpose(0, 1))
autocorrelation_plot(ry2.transpose(0, 1))
