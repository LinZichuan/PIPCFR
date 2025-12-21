import torch
import torch.nn as nn
from torch import optim
import math
import sys
import numpy as np
from torch.autograd import Function
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from torch.utils.data import Dataset, DataLoader


def init_weights(m):
    if isinstance(m, nn.Linear):
        stdv = 1 / math.sqrt(m.weight.size(1))
        torch.nn.init.normal_(m.weight, mean=0.0, std=stdv)
        # torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


def sigmod2(y):
    # y = torch.clamp(0.995 / (1.0 + torch.exp(-y)) + 0.0025, 0, 1)
    # y = torch.clamp(y, -16, 16)
    y = torch.sigmoid(y)
    # y = 0.995 / (1.0 + torch.exp(-y)) + 0.0025

    return y


def safe_sqrt(x):
    ''' Numerically safe version of Pytoch sqrt '''
    return torch.sqrt(torch.clip(x, 1e-9, 1e+9))


class ShareNetwork(nn.Module):
    def __init__(self, input_dim, share_dim, base_dim, cfg, device):
        super(ShareNetwork, self).__init__()
        if cfg.BatchNorm1d == 'true':
            print("use BatchNorm1d")
            self.DNN = nn.Sequential(

                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate),
                nn.Linear(share_dim, share_dim),
                # nn.BatchNorm1d(share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate),
                nn.Linear(share_dim, base_dim),
                # nn.BatchNorm1d(base_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate)
            )
        else:
            print("No BatchNorm1d")
            self.DNN = nn.Sequential(
                nn.Linear(input_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate),
                nn.Linear(share_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate),
                nn.Linear(share_dim, base_dim),
                nn.ELU(),
            )

        self.DNN.apply(init_weights)
        self.cfg = cfg
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        h_rep = self.DNN(x)
        if self.cfg.normalization == "divide":
            h_rep_norm = h_rep / safe_sqrt(torch.sum(torch.square(h_rep), dim=1, keepdim=True))
        else:
            h_rep_norm = 1.0 * h_rep
        return h_rep_norm


class BaseModel(nn.Module):
    def __init__(self, base_dim, cfg):
        super(BaseModel, self).__init__()
        self.DNN = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=cfg.do_rate),
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=cfg.do_rate),
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=cfg.do_rate)
        )
        self.DNN.apply(init_weights)

    def forward(self, x):
        logits = self.DNN(x)
        return logits


class PrpsyNetwork(nn.Module):
    """propensity network"""

    def __init__(self, base_dim, cfg):
        super(PrpsyNetwork, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.logitLayer.apply(init_weights)

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        return p


class Mu0Network(nn.Module):
    def __init__(self, base_dim, cfg):
        super(Mu0Network, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        # return self.relu(p)
        return p


class Mu1Network(nn.Module):
    def __init__(self, base_dim, cfg):
        super(Mu1Network, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        # return self.relu(p)
        return p


class TauNetwork(nn.Module):
    """pseudo tau network"""

    def __init__(self, base_dim, cfg):
        super(TauNetwork, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        tau_logit = self.logitLayer(inputs)
        # return self.tanh(p)
        return tau_logit


class PIPCFR(nn.Module):

    def __init__(self, prpsy_network: PrpsyNetwork, \
                 mu1_network: Mu1Network, mu0_network: Mu0Network, tau_network: TauNetwork, shareNetwork: ShareNetwork,
                 notshareNetwork: ShareNetwork,
                 next_shareNetwork: ShareNetwork, hs_mu1_network: Mu1Network, hs_mu0_network: Mu0Network,
                 all_mu1_network: Mu1Network, all_mu0_network: Mu0Network, all_prpsy_network: PrpsyNetwork,
                 cfg, device, tarreg=False, freeze_all_prpsy_network=None, share_rep=False, prpsy_shareNetwork: ShareNetwork=None, y_scaler=None):
        super(PIPCFR, self).__init__()
        # self.feature_extractor = feature_extractor
        self.shareNetwork = shareNetwork.to(device)
        self.notshareNetwork = notshareNetwork.to(device)
        self.prpsy_network = prpsy_network.to(device)
        self.mu1_network = mu1_network.to(device)
        self.mu0_network = mu0_network.to(device)
        self.tau_network = tau_network.to(device)

        self.next_shareNetwork = next_shareNetwork.to(device)
        self.hs_mu1_network = hs_mu1_network.to(device)
        self.hs_mu0_network = hs_mu0_network.to(device)
        self.all_mu1_network = all_mu1_network.to(device)
        self.all_mu0_network = all_mu0_network.to(device)
        self.all_prpsy_network = all_prpsy_network.to(device)
        if freeze_all_prpsy_network is not None:
            self.freeze_all_prpsy_network = freeze_all_prpsy_network.to(device)
        else:
            self.freeze_all_prpsy_network = self.all_prpsy_network

        if prpsy_shareNetwork is not None:
            print("prpsy_shareNetwork is not None")
            self.prpsy_shareNetwork = prpsy_shareNetwork.to(device)
        else:
            self.prpsy_shareNetwork = self.shareNetwork

        self.cfg = cfg
        self.tarreg = tarreg
        if self.tarreg:
            self.epsilon = nn.Linear(in_features=1, out_features=1)
            self.all_epsilon = nn.Linear(in_features=1, out_features=1)
        self.device = device
        self.share_rep = share_rep
        self.y_scaler = y_scaler
        self.to(device)

    def forward(self, inputs, next_inputs):
        shared_h = self.shareNetwork(inputs)

        p_prpsy_logit_main = self.prpsy_network(shared_h)
        p_prpsy_main = (torch.sigmoid(p_prpsy_logit_main) + 0.01) / 1.02

        next_share_h = self.next_shareNetwork(next_inputs)

        # propensity output_logit
        prpsy_share_h = self.prpsy_shareNetwork(inputs)
        p_prpsy_logit = self.prpsy_network(prpsy_share_h)

        # p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.05, 0.95)
        # p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.001, 0.999)
        p_prpsy = (torch.sigmoid(p_prpsy_logit) + 0.01) / 1.02

        # logit for mu1, mu0
        mu1_logit = self.mu1_network(shared_h)
        mu0_logit = self.mu0_network(shared_h)
        # mu1_logit = self.outcome_norm(self.mu1_network(shared_h))
        # mu0_logit = self.outcome_norm(self.mu0_network(shared_h))

        # pseudo tau
        tau_logit = self.tau_network(shared_h)

        p_mu1 = sigmod2(mu1_logit)
        p_mu0 = sigmod2(mu0_logit)
        p_h1 = p_mu1  # Refer to the naming in TARnet/CFR
        p_h0 = p_mu0  # Refer to the naming in TARnet/CFR

        # entire space
        p_estr = torch.mul(p_prpsy, p_h1)
        p_i_prpsy = 1 - p_prpsy
        p_escr = torch.mul(p_i_prpsy, p_h0)

        if self.share_rep:
            final_shared_h = shared_h.detach()
        else:
            final_shared_h = self.notshareNetwork(inputs)

        p_prpsy_logit_hs = self.prpsy_network(shared_h)
        p_prpsy_hs = (torch.sigmoid(p_prpsy_logit_hs) + 0.01) / 1.02

        all_mu1_logit = self.all_mu1_network(torch.cat((final_shared_h, next_share_h), dim=1))
        all_mu0_logit = self.all_mu0_network(torch.cat((final_shared_h, next_share_h), dim=1))
        # all_mu1_logit = self.all_mu1_network(final_shared_h)
        # all_mu0_logit = self.all_mu0_network(final_shared_h)
        hs_mu1_logit = self.hs_mu1_network(next_share_h)
        hs_mu0_logit = self.hs_mu0_network(next_share_h)

        all_p_mu1 = sigmod2(all_mu1_logit)
        all_p_mu0 = sigmod2(all_mu0_logit)
        hs_p_mu1 = sigmod2(hs_mu1_logit)
        hs_p_mu0 = sigmod2(hs_mu0_logit)

        all_prpsy_nograd_logit = self.all_prpsy_network(torch.cat((prpsy_share_h.detach(), next_share_h.detach()), dim=1))
        all_prpsy_logit = self.freeze_all_prpsy_network(torch.cat((prpsy_share_h.detach(), next_share_h), dim=1))

        # all_p_prpsy_nograd = torch.clip(sigmod2(all_prpsy_nograd_logit), 0.001, 0.999)
        # all_p_prpsy = torch.clip(sigmod2(all_prpsy_logit), 0.001, 0.999)
        all_p_prpsy_nograd = (sigmod2(all_prpsy_nograd_logit) + 0.01) / 1.02
        all_p_prpsy = (sigmod2(all_prpsy_logit) + 0.01) / 1.02

        if self.tarreg:
            eps = self.epsilon(torch.ones_like(p_prpsy)[:, 0:1])
            eps_item_1 = 1 / p_prpsy
            mu1_logit_pert = mu1_logit + eps * eps_item_1
            eps_item_0 = - 1 / (1 - p_prpsy)
            mu0_logit_pert = mu0_logit + eps * eps_item_0

            all_eps = self.all_epsilon(torch.ones_like(p_prpsy)[:, 0:1])
            all_eps_item_1 = 1 / p_prpsy
            all_mu1_logit_pert = all_mu1_logit + all_eps * all_eps_item_1
            all_eps_item_0 = - 1 / (1 - p_prpsy)
            all_mu0_logit_pert = all_mu0_logit + all_eps * all_eps_item_0

            return p_prpsy_logit, p_estr, p_escr, tau_logit, mu1_logit, mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h, final_shared_h, \
                all_mu1_logit, all_mu0_logit, hs_mu1_logit, hs_mu0_logit, all_prpsy_nograd_logit, all_prpsy_logit, \
                all_p_mu1, all_p_mu0, hs_p_mu1, hs_p_mu0, all_p_prpsy_nograd, all_p_prpsy, \
                eps, mu1_logit_pert, mu0_logit_pert, all_eps, all_mu1_logit_pert, all_mu0_logit_pert

        return p_prpsy_logit, p_estr, p_escr, tau_logit, mu1_logit, mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h, final_shared_h, \
            all_mu1_logit, all_mu0_logit, hs_mu1_logit, hs_mu0_logit, all_prpsy_nograd_logit, all_prpsy_logit, \
            all_p_mu1, all_p_mu0, hs_p_mu1, hs_p_mu0, all_p_prpsy_nograd, all_p_prpsy


class PIPCFR_on_DRCFR(nn.Module):
    def __init__(self, prpsy_network: PrpsyNetwork, \
                 mu1_network: Mu1Network, mu0_network: Mu0Network, shareNetwork: ShareNetwork,
                 DRCFR_network, next_shareNetwork: ShareNetwork, all_prpsy_network: PrpsyNetwork,
                 hs_mu1_network: Mu1Network, hs_mu0_network: Mu0Network,
                 device, freeze_all_prpsy_network=None, prpsy_shareNetwork: ShareNetwork=None, y_scaler=None):
        super(PIPCFR_on_DRCFR, self).__init__()
        self.shareNetwork = shareNetwork.to(device)
        self.prpsy_network = prpsy_network.to(device)
        self.mu1_network = mu1_network.to(device)
        self.mu0_network = mu0_network.to(device)
        self.hs_mu1_network = hs_mu1_network.to(device)
        self.hs_mu0_network = hs_mu0_network.to(device)
        self.DRCFR_network = DRCFR_network.to(device)
        self.next_shareNetwork = next_shareNetwork.to(device)
        self.all_prpsy_network = all_prpsy_network.to(device)
        if freeze_all_prpsy_network is not None:
            self.freeze_all_prpsy_network = freeze_all_prpsy_network.to(device)
        else:
            self.freeze_all_prpsy_network = self.all_prpsy_network

        if prpsy_shareNetwork is not None:
            print("prpsy_shareNetwork is not None")
            self.prpsy_shareNetwork = prpsy_shareNetwork.to(device)
        else:
            self.prpsy_shareNetwork = self.shareNetwork
        self.tarreg = False
        self.device = device
        self.y_scaler = y_scaler

    def forward(self, inputs, next_inputs):
        shared_h = self.shareNetwork(inputs)

        next_share_h = self.next_shareNetwork(next_inputs)

        prpsy_share_h = self.prpsy_shareNetwork(inputs)
        p_prpsy_logit = self.prpsy_network(prpsy_share_h)
        p_prpsy = (torch.sigmoid(p_prpsy_logit) + 0.01) / 1.02

        hs_mu1_logit = self.hs_mu1_network(next_share_h)
        hs_mu0_logit = self.hs_mu0_network(next_share_h)
        hs_p_mu1 = sigmod2(hs_mu1_logit)
        hs_p_mu0 = sigmod2(hs_mu0_logit)

        p_prpsy_logit_hs, p_prpsy_hs, all_mu1_logit, all_mu0_logit, all_p_tau, \
        all_p_yf, all_p_ycf, final_gamma, final_delta, final_upsilon = self.DRCFR_network(inputs, next_share_h)

        p_prpsy_logit_main = self.prpsy_network(final_delta + final_upsilon)
        p_prpsy_main = (torch.sigmoid(p_prpsy_logit_main) + 0.01) / 1.02

        mu1_logit = self.mu1_network(final_gamma + final_delta)
        mu0_logit = self.mu0_network(final_gamma + final_delta)

        p_mu1 = sigmod2(mu1_logit)
        p_mu0 = sigmod2(mu0_logit)
        p_h1 = p_mu1
        p_h0 = p_mu0

        all_prpsy_nograd_logit = self.all_prpsy_network(torch.cat((prpsy_share_h.detach(), next_share_h.detach()), dim=1))
        all_prpsy_logit = self.freeze_all_prpsy_network(torch.cat((prpsy_share_h.detach(), next_share_h), dim=1))

        all_p_prpsy_nograd = (sigmod2(all_prpsy_nograd_logit) + 0.01) / 1.02
        all_p_prpsy = (sigmod2(all_prpsy_logit) + 0.01) / 1.02

        return p_prpsy_logit, p_prpsy, p_prpsy_logit_main, p_prpsy_main, p_prpsy_logit_hs, p_prpsy_hs, \
            shared_h, final_gamma, final_delta, final_upsilon, mu1_logit, mu0_logit, p_mu1, p_mu0, p_h1, p_h0, \
            all_mu1_logit, all_mu0_logit, all_prpsy_nograd_logit, all_prpsy_logit, all_p_prpsy_nograd, all_p_prpsy, \
            hs_mu1_logit, hs_mu0_logit, hs_p_mu1, hs_p_mu0 




class EarlyStopper:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class TransformerNetwork(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, output_dim, dropout, device):
        super(TransformerNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = self._generate_positional_encoding(hidden_dim, 10000)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.device = device
        self.to(device)
        self.positional_encoding = self.positional_encoding.to(device)

    def _generate_positional_encoding(self, hidden_dim, max_len):
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def forward(self, x):
        x = x.to(self.device)
        x = self.embedding(x) + self.positional_encoding[:x.size(0), :]
        x = self.transformer_encoder(x)
        x = self.output_layer(x)
        x = torch.mean(x, dim=1)  # 对时间步维度进行平均池化
        return x


class MyPipeline:
    def __init__(self, steps=[]):
        if not steps:
            self.pipeline = None
        elif len(steps) == 1:
            self.pipeline = steps[0][1]
        else:
            self.pipeline = Pipeline(steps=steps)

    def fit(self, X_train):
        if self.pipeline is not None:
            self.pipeline.fit(X_train)

    def fit_transform(self, X_train):
        if self.pipeline is not None:
            return self.pipeline.fit_transform(X_train)
        return X_train

    def transform(self, X_train):
        if self.pipeline is not None:
            return self.pipeline.transform(X_train)
        return X_train


def make_base_regressor(model_name, input_dim, device, cfg, binary, epochs):
    if '_linear' in model_name.lower():
        return Linear_model(binary=binary, epochs=epochs)
    if "_mlp" in model_name.lower():
        return MLP(input_dim, cfg.share_dim, device, cfg.lr, cfg.l2, cfg.do_rate, binary, epochs=epochs)


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, lr, weight_decay, do_rate=0.1, binary=False, epochs=100):
        super(MLP, self).__init__()
        self.DNN = nn.Sequential(

            # nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=do_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=do_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=do_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=do_rate),
            nn.Linear(hidden_dim, 1),
        )
        self.binary = binary
        if binary:
            self.output_layer = nn.Sigmoid()
        else:
            self.output_layer = nn.Identity()
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.epochs = epochs
        self.to(device)

    def forward(self, x):
        output = self.DNN(x)
        output = self.output_layer(output)
        return output

    def fit(self, inputs, labels):

        print("inputs shape:", inputs.shape)
        print("labels shape:", labels.shape)
        inputs = torch.from_numpy(inputs).float().to(self.device)
        labels = torch.from_numpy(labels.reshape((-1, 1))).float().to(self.device)

        datasets = MyDataset(inputs, labels)
        dataloader = DataLoader(datasets, batch_size=128, shuffle=True)

        loss_fn = nn.BCELoss() if self.binary else nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.train()

        best_loss = float('inf')
        epochs_without_improvement = 0
        patience = 5

        for epoch in range(self.epochs):
            epoch_loss = 0
            for i, (batch_inputs, batch_labels) in enumerate(dataloader):
                predictions = self.forward(batch_inputs)
                loss = loss_fn(predictions, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                print("Epoch {}, batch {}: Loss = {}".format(epoch, i, loss))

            avg_epoch_loss = epoch_loss / len(dataloader)
            print("Epoch {}: Average Loss = {}".format(epoch, avg_epoch_loss))

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Early stopping triggered after {} epochs without improvement.".format(patience))
                break

    def predict(self, inputs):
        self.eval()
        inputs = torch.from_numpy(inputs).float().to(self.device)
        outputs = self.forward(inputs)
        return outputs.cpu().detach().numpy()


class Linear_model():
    def __init__(self, binary=False, epochs=100):
        self.binary = binary
        if binary:
            self.reg = LogisticRegression(max_iter=epochs)
        else:
            # self.reg = LinearRegression()
            self.reg = Ridge(alpha=1.0, max_iter=epochs)
        self.epochs = epochs

    def fit(self, inputs, labels):
        labels = labels.flatten()
        self.reg.fit(inputs, labels)

    def predict(self, inputs):
        if self.binary:
            return self.reg.predict_proba(inputs)[:, 1].reshape((-1, 1))
        return self.reg.predict(inputs).reshape((-1, 1))
    

class DRCFR(nn.Module):
    def __init__(self, input_dim, share_dim, base_dim, cfg, device, next_dim=0):
        super(DRCFR, self).__init__()
        # 三个表示学习网络
        self.gamma_net = ShareNetwork(input_dim, share_dim, base_dim, cfg, device)
        self.delta_net = ShareNetwork(input_dim, share_dim, base_dim, cfg, device)
        self.upsilon_net = ShareNetwork(input_dim, share_dim, base_dim, cfg, device)
        
        # 处理分配网络(使用gamma和delta)
        self.prpsy_network = PrpsyNetwork(2 * base_dim, cfg)
        
        # 结果预测网络(使用delta和upsilon)
        self.mu1_network = Mu1Network(2 * base_dim+next_dim, cfg)
        self.mu0_network = Mu0Network(2 * base_dim+next_dim, cfg)
        
        self.cfg = cfg
        self.device = device
        self.to(device)

    def forward(self, x, next_share_h=None):
        # 提取三个因子
        gamma = self.gamma_net(x)
        delta = self.delta_net(x)
        upsilon = self.upsilon_net(x)
        
        # 预测处理分配概率
        t_input = torch.cat([gamma, delta], dim=1)
        p_prpsy_logit = self.prpsy_network(t_input)
        p_prpsy = (torch.sigmoid(p_prpsy_logit) + 0.01) / 1.02
        
        # 预测潜在结果
        if next_share_h is not None:
            y_input = torch.cat([delta, upsilon, next_share_h.reshape(delta.shape[0], -1)], dim=1)
        else:
            y_input = torch.cat([delta, upsilon], dim=1)
        mu1_logit = self.mu1_network(y_input)
        mu0_logit = self.mu0_network(y_input)
        
        # p_mu1 = sigmod2(mu1_logit)
        # p_mu0 = sigmod2(mu0_logit)

        p_mu1 = mu1_logit
        p_mu0 = mu0_logit
        
        # 计算tau
        p_tau = p_mu1 - p_mu0
        
        # 计算factual和counterfactual预测
        p_yf = p_mu1 * p_prpsy + p_mu0 * (1 - p_prpsy)
        p_ycf = p_mu0 * p_prpsy + p_mu1 * (1 - p_prpsy)
        
        return p_prpsy_logit, p_prpsy, p_mu1, p_mu0, p_tau, p_yf, p_ycf, gamma, delta, upsilon
    
class PrpsyNetworkTrainer:
    def __init__(self, base_dim, cfg, lr=0.001, device='auto'):
        """
        Propensity Network Trainer
        Args:
            base_dim: 输入特征维度
            cfg: BaseModel配置参数
            lr: 学习率 (default: 0.001)
            device: 训练设备 ('cuda', 'cpu' 或 'auto')
        """
        # 自动检测设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if device == 'auto' else torch.device(device)
        
        # 初始化模型
        self.model = PrpsyNetwork(base_dim, cfg).to(self.device)
        
        # 设置优化器和损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()  # 自动包含sigmoid

    def fit(self, x_train, t_train, epochs=10, batch_size=32, verbose=True):
        """
        训练网络
        Args:
            x_train: 训练数据 (numpy array 或 torch.Tensor)
            t_train: 训练标签 (numpy array 或 torch.Tensor)
            epochs: 训练轮次 (default: 10)
            batch_size: 批大小 (default: 32)
            verbose: 是否打印训练信息 (default: True)
        """
        # 转换数据为张量
        x_tensor = torch.from_numpy(x_train).float().to(self.device)
        t_tensor = torch.from_numpy(t_train.reshape((-1, 1))).float().to(self.device)

        # 创建数据加载器
        datasets = MyDataset(x_tensor, t_tensor)
        dataloader = DataLoader(datasets, batch_size=128, shuffle=True)

        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, targets in dataloader:
                # 数据迁移到设备
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # 反向传播
                loss.backward()
                self.optimizer.step()

                # 记录损失
                epoch_loss += loss.item() * inputs.size(0)

            # 计算平均损失
            epoch_loss /= len(dataloader.dataset)
            
            if verbose:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

    def predict(self, x):
        """
        预测概率
        Args:
            x: 输入数据 (numpy array 或 torch.Tensor)
        Returns:
            预测概率 (numpy array)
        """
        self.model.eval()
        with torch.no_grad():
            # 转换输入数据
            x_tensor = torch.as_tensor(x, dtype=torch.float32).to(self.device)
            
            # 预测并应用sigmoid
            logits = self.model(x_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()
        return probabilities.squeeze()  # 去除多余维度
