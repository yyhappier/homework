import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, PairNorm  # , GCNConv
from torch.optim import Adam
from torch_geometric.utils import dropout_edge, negative_sampling
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
import json
from networkx.readwrite import json_graph
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt


def encode_label(labels):
    classes = set(labels)
    classes_dict = {c: i for i, c in enumerate(classes)}
    labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels


def load_cite_data(path="../data/cora/", dataset="cora", task='node'):
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    # 得到论文单词组成，也就是节点特征
    x = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 得到论文的类别，也就是节点标签。 把字符类型标签映射为类别
    y = encode_label(idx_features_labels[:, -1])
    num_classes = torch.tensor(np.max(y) + 1)

    # 把节点名称映射为下标，方便后面提取边
    idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.dtype(str))

    # 提取边
    edge_index = [[], []]
    for i in range(edges_unordered.shape[0]):
        try:  # 判断边的两端是否在节点列表中，如果不在就去掉这条边
            start_idx = idx_map[edges_unordered[i, 0]]
            end_idx = idx_map[edges_unordered[i, 1]]
        except KeyError:
            continue
        edge_index[0].append(start_idx)
        edge_index[1].append(end_idx)
    edge_index = np.array(edge_index, dtype=np.int32)

    # 转换为 tensor 类型
    x = torch.FloatTensor(x)
    y = torch.LongTensor(y)
    edge_index = torch.LongTensor(edge_index)

    data_ = Data(x=x, edge_index=edge_index, y=y)
    if task == 'node':  # 节点分类任务
        # 随机划分训练集、验证集和测试集
        # 训练集: 验证集: 测试集 = 0.6: 0.2: 0.2
        mask = np.random.permutation(idx.shape[0])
        train_mask = mask[: int(idx.shape[0] * 0.6)]
        val_mask = mask[int(idx.shape[0] * 0.6): int(idx.shape[0] * 0.8)]
        test_mask = mask[int(idx.shape[0] * 0.8):]

        # 把数据转换为合适的类型，并且封装到图数据的类型
        train_mask = torch.LongTensor(train_mask)
        val_mask = torch.LongTensor(val_mask)
        test_mask = torch.LongTensor(test_mask)

        data_.train_mask = train_mask
        data_.val_mask = val_mask
        data_.test_mask = test_mask
        data_.num_classes = num_classes
    else:  # 链路预测任务
        # 随机划分边为训练集，验证集和测试集。训练集无负样本。图为有向图。
        # 训练集: 验证集: 测试集 = 0.6: 0.2: 0.2
        transform = RandomLinkSplit(is_undirected=False, num_val=0.2, num_test=0.2,
                                    add_negative_train_samples=False)
        train_data, val_data, test_data = transform(data_)
        data_.train_pos_edge_index = train_data.edge_label_index
        data_.val_edge_index = val_data.edge_label_index
        data_.val_edge_label = val_data.edge_label
        data_.test_edge_index = test_data.edge_label_index
        data_.test_edge_label = test_data.edge_label

    return data_


def load_ppi_data(path="../data/ppi/", dataset="ppi", task='node'):
    prefix = path + dataset
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    feats = np.load(prefix + "-feats.npy")
    class_map = json.load(open(prefix + "-class_map.json"))

    x = feats
    x = torch.FloatTensor(x)
    edge_index = np.array(G.edges()).T
    edge_index = torch.LongTensor(edge_index)
    str_nodes = list(map(str, G.nodes))
    y = np.array(list(map(class_map.get, str_nodes)))
    y = torch.FloatTensor(y)  # 这里需要使用 Float 类型，后面计算 BCELoss 时需要输入为 Float 类型
    data_ = Data(x=x, edge_index=edge_index, y=y)

    if task == 'node':  # 节点分类任务
        num_classes = torch.tensor(y.size(1))
        train_mask = []
        val_mask = []
        test_mask = []
        # 这里数据集中已经划分好训练集、验证集和测试集。
        for node in G.nodes():
            if G.nodes()[node]['val']:
                val_mask.append(node)
            elif G.nodes()[node]['test']:
                test_mask.append(node)
            else:
                train_mask.append(node)
        train_mask, val_mask = torch.LongTensor(
            train_mask), torch.LongTensor(val_mask)
        test_mask = torch.LongTensor(test_mask)

        data_.train_mask = train_mask
        data_.val_mask = val_mask
        data_.test_mask = test_mask
        data_.num_classes = num_classes
    else:  # 链路预测任务
        # 随机划分边为训练集，验证集和测试集。训练集无负样本。图为无向图。
        # 训练集: 验证集: 测试集 = 0.6: 0.2: 0.2
        transform = RandomLinkSplit(is_undirected=True, num_val=0.2, num_test=0.2,
                                    add_negative_train_samples=False)
        train_data, val_data, test_data = transform(data_)
        data_.train_pos_edge_index = train_data.edge_label_index
        data_.val_edge_index = val_data.edge_label_index
        data_.val_edge_label = val_data.edge_label
        data_.test_edge_index = test_data.edge_label_index
        data_.test_edge_label = test_data.edge_label
    return data_


def load_data(path, dataset, task='node'):
    print('Loading {} dataset...'.format(dataset))
    if dataset != "ppi":
        return load_cite_data(path, dataset, task)
    else:
        return load_ppi_data(path, dataset, task)


def accuracy(output, labels, dataset, task='node'):
    if task == 'node':
        if dataset != "ppi":  # 正确率
            preds = output.max(dim=1)[1].type_as(labels)
            correct = preds.eq(labels).double()
            correct = correct.sum()
            score = correct.item() / len(labels)
        else:  # Micro F1 score
            predict = output.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            # predict = np.where(predict > 0.5, 1, 0)
            predict = np.where(predict > 0., 1, 0)
            score = f1_score(labels, predict, average='micro')
    else:  # AUC 指标
        predict = output.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        score = roc_auc_score(labels, predict)
    return score


def criterion(output, target, dataset, task='node'):
    # print(output.dtype, target.dtype)
    if task == 'node':
        if dataset != 'ppi':
            loss = nn.CrossEntropyLoss()(output, target)
        else:
            output = nn.Sigmoid()(output)
            loss = nn.BCELoss()(output, target)
    else:  # 二分类任务
        loss = nn.functional.binary_cross_entropy_with_logits(output, target)

    return loss


def plot_loss(loss_list, dataset, task):
    plt.figure()
    train_loss = loss_list['train']
    val_loss = loss_list['val']

    plt.plot(train_loss, c="red", label="train_loss")
    plt.plot(val_loss, c="blue", label="val_loss")

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"Training and Validation Loss of {task} on {dataset} in Each Epoch")
    plt.savefig(f"./fig/{task}_{dataset}_loss.png")


def _add_self_loops(edge_index, num_nodes):
    """
    添加自环
    """
    loop_index = torch.arange(
        0, num_nodes, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index


def _degree(index, num_nodes, dtype):
    """
    计算每个节点的度
    """
    out = torch.zeros((num_nodes), dtype=dtype, device=index.device)
    return out.scatter_add_(0, index, out.new_ones((index.size(0))))


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, add_self_loops=True):
        # "mean" aggregation (Step 5).
        super(GCNConv, self).__init__(aggr='mean')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.add_self_loops = add_self_loops

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        if self.add_self_loops:
            edge_index = _add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        col_deg = _degree(col, x.size(0), dtype=x.dtype)
        row_deg = _degree(row, x.size(0), dtype=x.dtype)
        col_deg_inv_sqrt = col_deg.pow(-0.5)
        row_deg_inv_sqrt = row_deg.pow(-0.5)
        norm = row_deg_inv_sqrt[row] * col_deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


class NodeNet(torch.nn.Module):
    """Network for node classification
    """

    def __init__(self, node_features, hidden_features, num_layers, num_classes, add_self_loops=True,
                 pair_norm=False, drop_edge=False, act_fn='prelu', dataset='cora'):
        super(NodeNet, self).__init__()
        self.pair_norm = pair_norm
        self.drop_edge = drop_edge
        self.convs = nn.ModuleList()
        self.dataset = dataset
        for i in range(num_layers):
            in_channels = hidden_features
            out_channels = hidden_features
            if i == 0:
                in_channels = node_features

            if i == num_layers - 1:
                out_channels = num_classes

            self.convs.append(GCNConv(in_channels=in_channels, out_channels=out_channels,
                                      add_self_loops=add_self_loops))

        if pair_norm:
            self.pn = PairNorm()

        if act_fn == 'prelu':
            self.act_fn = nn.PReLU()
        elif act_fn == 'relu':
            self.act_fn = nn.ReLU()
        elif act_fn == 'softplus':
            self.act_fn = nn.Softplus()
        else:
            self.act_fn = nn.Tanh()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.pair_norm:
                x = self.pn(x)
            if self.drop_edge:
                edge_index = dropout_edge(edge_index=edge_index, p=0.2)[0]
            if self.dataset == 'ppi':
                x = self.act_fn(x)
            else:
                if i != len(self.convs) - 1:
                    x = self.act_fn(x)
        return x


class NodeClassification(object):
    def __init__(self, device="cuda", dataset="cora", path="../data/cora/"):
        super(NodeClassification, self).__init__()
        self.net = None
        self.device = torch.device(device)
        self.data = None
        self.path = path
        self.dataset = dataset
        self.loss_list = {"train": [], "val": []}

    def train(self, patience=3, epochs=10, lr=2e-5, hidden_features=16, num_layers=2, add_self_loops=True,
              pair_norm=False, drop_edge=False, test=False, act_fn='prelu'):
        self.data = load_data(
            path=self.path, dataset=self.dataset, task='node')
        num_classes = self.data['num_classes'].item()
        self.data = self.data.to(self.device)
        self.net = NodeNet(node_features=self.data.num_features, hidden_features=hidden_features,
                           num_classes=num_classes, num_layers=num_layers, add_self_loops=add_self_loops,
                           pair_norm=pair_norm, drop_edge=drop_edge, act_fn=act_fn, dataset=self.dataset)
        total_params = sum([param.nelement()
                           for param in self.net.parameters()])
        print(f">>> total params: {total_params}")
        self.net.to(self.device)
        optimizer = Adam(self.net.parameters(), lr=lr)
        best_model_path = f"./model/{self.dataset}_node_best.pth"
        delay = 0
        best_val_loss = np.inf
        best_val_score = -1
        for epoch in range(epochs):
            self.net.train()
            optimizer.zero_grad()
            out = self.net(self.data)
            train_loss = criterion(
                out[self.data.train_mask], self.data.y[self.data.train_mask], self.dataset)
            self.loss_list['train'].append(train_loss.item())
            # train_score = accuracy(out[self.data.train_mask], self.data.y[self.data.train_mask], self.dataset)
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.net.eval()
                val_loss = criterion(
                    out[self.data.val_mask], self.data.y[self.data.val_mask], self.dataset).item()
                self.loss_list['val'].append(val_loss)

                val_score = accuracy(
                    out[self.data.val_mask], self.data.y[self.data.val_mask], self.dataset)
                # if (epoch % 10) == 0:
                #     print(f"epoch: {epoch}, train_loss: {train_loss.item():7.5f}, train_score: {train_score:7.5f}, "
                #           f"val_loss: {val_loss:7.5f}, val_score: {val_score:7.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_score = val_score
                torch.save(self.net, best_model_path)
                delay = 0
            else:
                delay += 1
                if delay > patience:
                    break

        print(">>> Finished training")
        plot_loss(self.loss_list, 'Node Classification', self.dataset)
        print(">>> Finished plot loss")
        print(f"best_val_loss: {best_val_loss:7.4f}, best_val_score: {best_val_score:7.4f} \n"
              f"{best_val_loss:7.4f} | {best_val_score:7.4f} |")

        if test:  # whether test on test dataset
            self.net = torch.load(best_model_path)
            self.net.to(self.device)
            with torch.no_grad():
                self.net.eval()
                out = self.net(self.data)
                test_loss = criterion(
                    out[self.data.test_mask], self.data.y[self.data.test_mask], self.dataset).item()
                test_score = accuracy(
                    out[self.data.test_mask], self.data.y[self.data.test_mask], self.dataset)
                print(f"test_score {test_score:7.4f}")


class LinkNet(torch.nn.Module):
    """Network for link prediction
    """

    def __init__(self, node_features, hidden_features, num_layers, add_self_loops=True, pair_norm=False,
                 drop_edge=False, act_fn='prelu') -> None:
        super(LinkNet, self).__init__()
        self.pair_norm = pair_norm
        self.drop_edge = drop_edge
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(
                    GCNConv(in_channels=node_features, out_channels=hidden_features, add_self_loops=add_self_loops))
            else:
                self.convs.append(
                    GCNConv(in_channels=hidden_features, out_channels=hidden_features, add_self_loops=add_self_loops))
        if self.pair_norm:
            self.pn = PairNorm()

        if act_fn == 'prelu':
            self.act_fn = nn.PReLU()
        elif act_fn == 'relu':
            self.act_fn = nn.ReLU()
        elif act_fn == 'softplus':
            self.act_fn = nn.Softplus()
        else:
            self.act_fn = nn.Tanh()

    def encode(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.pair_norm:
                x = self.pn(x)
            if self.drop_edge:
                edge_index = dropout_edge(edge_index=edge_index, p=0.2)[0]
            if i != len(self.convs) - 1:
                x = self.act_fn(x)
        return x

    def decode(self, z, edge_index):
        # edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)  #[2, E]
        # element-wise 乘法
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


class LinkPrediction(object):
    def __init__(self, device="cuda", dataset="cora", path="../data/cora/") -> None:
        super(LinkPrediction, self).__init__()
        self.net = None
        self.device = torch.device(device)
        self.data = None
        self.path = path
        self.dataset = dataset
        self.loss_list = {"train": [], "val": []}

    def get_link_labels(self, pos_edge_index, neg_edge_index):
        num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(num_links, dtype=torch.float)
        link_labels[:pos_edge_index.size(1)] = 1
        return link_labels

    def train(self, patience=3, epochs=10, lr=2e-5, hidden_features=16, num_layers=2, add_self_loops=True,
              pair_norm=False, drop_edge=False, test=False, act_fn='prelu'):
        self.data = load_data(self.path, self.dataset, 'link')
        self.data = self.data.to(self.device)
        self.net = LinkNet(self.data.num_features, hidden_features, num_layers, add_self_loops, pair_norm, drop_edge,
                           act_fn=act_fn)
        total_params = sum([param.nelement()
                           for param in self.net.parameters()])
        print(f">>> total params: {total_params}")
        self.net.to(self.device)
        optimizer = Adam(self.net.parameters(), lr=lr)
        best_model_path = f"./model/{self.dataset}_link_best.pth"
        delay = 0
        best_val_loss = np.inf
        best_val_score = -1
        for epoch in range(epochs):
            neg_edge_index = negative_sampling(edge_index=self.data.train_pos_edge_index,
                                               num_nodes=self.data.num_nodes,
                                               num_neg_samples=self.data.train_pos_edge_index.size(1))
            self.net.train()
            optimizer.zero_grad()
            z = self.net.encode(self.data.x, self.data.train_pos_edge_index)
            edge_index = torch.cat(
                [self.data.train_pos_edge_index, neg_edge_index], dim=-1)
            link_logits = self.net.decode(z, edge_index)
            link_labels = self.get_link_labels(
                self.data.train_pos_edge_index, neg_edge_index).to(self.data.x.device)

            train_loss = criterion(
                link_logits, link_labels, dataset=self.data, task='link')
            # train_score = accuracy(link_logits.sigmoid(), link_labels, self.dataset, 'link')
            self.loss_list['train'].append(train_loss.item())

            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.net.eval()

                z = self.net.encode(
                    self.data.x, self.data.train_pos_edge_index)
                edge_index = self.data.val_edge_index

                link_logits = self.net.decode(z, edge_index)
                link_probs = link_logits.sigmoid()
                link_labels = self.data.val_edge_label
                val_loss = criterion(
                    link_logits, link_labels, self.dataset, 'link').item()
                self.loss_list['val'].append(val_loss)
                val_score = accuracy(
                    link_probs, link_labels, self.dataset, 'link')
                # if (epoch % 10) == 0:
                # print(f"epoch: {epoch}, train_loss: {train_loss.item():7.5f}, train_score: {train_score:7.5f}, "
                #       f"val_loss: {val_loss:7.5f}, val_score: {val_score:7.5f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_score = val_score
                    torch.save(self.net, best_model_path)
                    delay = 0
                else:
                    delay += 1
                    if delay > patience:
                        break

        print(">>> Finished training")
        plot_loss(self.loss_list, 'Link Prediction', self.dataset)
        print(">>> Finished plot loss")
        print(f"best_val_loss: {best_val_loss:7.4f}, best_val_score: {best_val_score:7.4f} \n"
              f"{best_val_loss:7.4f} | {best_val_score:7.4f} |")
        if test:  # whether test on test dataset
            self.net = torch.load(best_model_path)
            self.net.to(self.device)
            with torch.no_grad():
                self.net.eval()
                z = self.net.encode(
                    self.data.x, self.data.train_pos_edge_index)
                edge_index = self.data.test_edge_index

                link_logits = self.net.decode(z, edge_index)
                link_probs = link_logits.sigmoid()
                link_labels = self.data.test_edge_label
                test_loss = criterion(
                    link_logits, link_labels, self.dataset, 'link')
                test_score = accuracy(
                    link_probs, link_labels, self.dataset, 'link')
                print(f"test_score {test_score:7.4f}")


act_fn = 'prelu'  # 激活函数
num_layers = 1  # GCN 层数
add_self_loops = True  # 是否添加自环
pair_norm = False  # 是否添加 pair_norm
drop_edge = True  # 是否去边
test = True  # 是否在训练结束后在测试集上测试模型

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# node classification task
model = NodeClassification(device=device, dataset="cora", path="./data/cora/")
model.train(patience=4, epochs=1000, lr=1e-3, hidden_features=64, num_layers=num_layers,
            add_self_loops=add_self_loops, pair_norm=pair_norm, drop_edge=drop_edge, test=test,
            act_fn=act_fn)
model = NodeClassification(device=device, dataset="citeseer", path="./data/citeseer/")
model.train(patience=4, epochs=500, lr=1e-3, hidden_features=64, num_layers=num_layers,
            add_self_loops=add_self_loops, pair_norm=pair_norm, drop_edge=drop_edge, test=test,
            act_fn=act_fn)
model = NodeClassification(device=device, dataset="ppi", path="./data/ppi/")
model.train(patience=4, epochs=500, lr=1e-1, hidden_features=128, num_layers=num_layers,
            add_self_loops=add_self_loops, pair_norm=pair_norm, drop_edge=drop_edge, test=test,
            act_fn=act_fn)

# link prediction task
model = LinkPrediction(device=device, dataset='cora', path="./data/cora/")
model.train(patience=4, epochs=500, lr=1e-3, hidden_features=64, num_layers=num_layers,
            add_self_loops=add_self_loops, pair_norm=pair_norm, drop_edge=drop_edge, test=test,
            act_fn=act_fn)
model = LinkPrediction(device=device, dataset="citeseer",path="./data/citeseer/")
model.train(patience=4, epochs=500, lr=1e-3, hidden_features=64, num_layers=num_layers,
            add_self_loops=add_self_loops, pair_norm=pair_norm, drop_edge=drop_edge, test=test,
            act_fn=act_fn)
model = LinkPrediction(device=device, dataset="ppi", path="./data/ppi/")
model.train(patience=4, epochs=500, lr=1e-3, hidden_features=64, num_layers=num_layers,
            add_self_loops=add_self_loops, pair_norm=pair_norm, drop_edge=drop_edge, test=test,
            act_fn=act_fn)
