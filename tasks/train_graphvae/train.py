import scipy.sparse as sp
import numpy as np
import torch
import time
import os
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import networkx as nx
import pickle
import sys
import tqdm
import hydra
from hydra.utils import to_absolute_path

from src.models import GATModelVAE


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# 这个函数的作用就是 返回一个稀疏矩阵的非0值坐标、非0值和整个矩阵的shape
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):  # 判断是否为coo_matrix类型
        sparse_mx = sparse_mx.tocoo()  # 返回稀疏矩阵的coo_matrix形式
    # 这个coo_matrix类型 其实就是系数矩阵的坐标形式：（所有非0元素 （row，col））根据row和col对应的索引对应非0元素在矩阵中的位置
    # 其他位置自动补0
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    # vstack 按垂直方向排列 再转置 则每一行的两个元素组成的元组就是非0元素的坐标
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    '''
    :param adj:
    :return:
    '''
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])  # 邻接矩阵加入自身信息，adj = adj + I
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())  # 节点的度矩阵
    # 正则化，D^{-0.5}(adj+I)D^{-0.5}
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def mask_test_edges(adj):
    '''
    构造train、val and test set
    function to build test set with 2% positive links
    remove diagonal elements
    :param adj:去除对角线元素的邻接矩阵
    :return:
    '''
    adj_triu = sp.triu(adj)  # 取出稀疏矩阵的上三角部分的非零元素，返回的是coo_matrix类型
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]  # 取除去节点自环的所有边（注意，由于adj_tuple仅包含原始邻接矩阵上三角的边，所以edges中的边虽然只记录了边<src,dis>，而不冗余记录边<dis,src>），shape=(边数,2)每一行记录一条边的起始节点和终点节点的编号
    edges_all = sparse_to_tuple(adj)[0]  # 取原始graph中的所有边，shape=(边数,2)每一行记录一条边的起始节点和终点节点的编号
    num_test = int(np.floor(edges.shape[0] / 50.))
    num_val = int(np.floor(edges.shape[0] / 50.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)  # 打乱all_edge_idx的顺序
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]  # edges是除去节点自环的所有边（因为数据集中的边都是无向的，edges只是存储了<src,dis>,没有存储<dis,src>），shape=(边数,2)每一行记录一条边的起始节点和终点节点的编号
    val_edges = edges[val_edge_idx]
    # np.vstack():垂直方向堆叠，np.hstack()：水平方向平铺
    # 删除test和val数据集，留下train数据集
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismemeber(a, b):
        # 判断随机生成的<a,b>这条边是否是已经真实存在的边，如果是，则返回True，否则返回False
        rows_close = np.all((a - b[:, None]) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    with tqdm.tqdm(range(len(test_edges))) as pbar:
        while len(test_edges_false) < len(test_edges):
            # test集中生成负样本边，即原始graph中不存在的边
            n_rnd = len(test_edges) - len(test_edges_false)
            # 随机生成
            rnd = np.random.randint(0, adj.shape[0], size=2 * n_rnd)
            idxs_i = rnd[:n_rnd]
            idxs_j = rnd[n_rnd:]
            for i in range(n_rnd):
                idx_i = idxs_i[i]
                idx_j = idxs_j[i]
                if idx_i == idx_j:
                    continue
                if ismemeber([idx_i, idx_j], edges_all):  # 如果随机生成的边<idx_i,idx_j>是原始graph中真实存在的边
                    continue
                if test_edges_false:  # 如果test_edges_false不为空
                    if ismemeber([idx_j, idx_i], np.array(test_edges_false)):  # 如果随机生成的边<idx_j,idx_i>是test_edges_false中已经包含的边
                        continue
                    if ismemeber([idx_i, idx_j], np.array(test_edges_false)):  # 如果随机生成的边<idx_i,idx_j>是test_edges_false中已经包含的边
                        continue
                test_edges_false.append([idx_i, idx_j])
                pbar.update()

    val_edge_false = []
    with tqdm.tqdm(range(len(val_edges))) as pbar:
        while len(val_edge_false) < len(val_edges):
            # val集中生成负样本边，即原始graph中不存在的边
            n_rnd = len(val_edges) - len(val_edge_false)
            rnd = np.random.randint(0, adj.shape[0], size=2 * n_rnd)
            idxs_i = rnd[:n_rnd]
            idxs_j = rnd[n_rnd:]
            for i in range(n_rnd):
                idx_i = idxs_i[i]
                idx_j = idxs_j[i]
                if idx_i == idx_j:
                    continue
                if ismemeber([idx_i, idx_j], train_edges):
                    continue
                if ismemeber([idx_j, idx_i], train_edges):
                    continue
                if ismemeber([idx_i, idx_j], val_edges):
                    continue
                if ismemeber([idx_j, idx_i], val_edges):
                    continue
                if val_edge_false:
                    if ismemeber([idx_j, idx_i], np.array(val_edge_false)):
                        continue
                    if ismemeber([idx_i, idx_j], np.array(val_edge_false)):
                        continue
                val_edge_false.append([idx_i, idx_j])
                pbar.update()

    # re-build adj matrix
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    # 这些边列表只包含一个方向的边（adj_train是矩阵，不是edge lists）
    return adj_train, train_edges, val_edges, val_edge_false, test_edges, test_edges_false


def vgae_loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    '''
    变分图自编码，损失函数包括两部分：
        1.生成图和原始图之间的距离度量
        2.节点表示向量分布和正态分布的KL散度
    '''
    # 负样本边的weight都为1，正样本边的weight都为pos_weight

    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


def gae_loss_function(preds, labels, norm, pos_weight):
    '''
    图自编码，损失函数是生成图和原始图之间的距离度量
    '''
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    return cost


def varga_loss_function(preds, labels, dis_real, dis_fake, mu, logvar, n_nodes, norm, pos_weight, device=torch.device('cpu')):
    # 对抗变分图正则化图自编码损失：生成和判别的loss
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    dis_real_loss = F.binary_cross_entropy_with_logits(dis_real, torch.ones(dis_real.shape).to(device))
    dis_fake_loss = F.binary_cross_entropy_with_logits(dis_fake, torch.zeros(dis_fake.shape).to(device))
    return cost + KLD + dis_real_loss + dis_fake_loss


def arga_loss_function(preds, labels, dis_real, dis_fake, norm, pos_weight, device=torch.device('cpu')):
    # 对抗图正则化图自编码损失：生成和判别的loss
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    dis_real_loss = F.binary_cross_entropy_with_logits(dis_real, torch.ones(dis_real.shape).to(device))
    dis_fake_loss = F.binary_cross_entropy_with_logits(dis_fake, torch.zeros(dis_fake.shape).to(device))
    return cost + dis_real_loss + dis_fake_loss


#
def real_load_data_with_features(path_to_dataset: str):
    dataset = pickle.load(open(path_to_dataset, 'rb'))
    num_nodes = max(max(dataset['pos_u']), max(dataset['pos_v'])) + 1
    adj = sp.coo_matrix((np.ones(shape=(len(dataset['pos_u']),)), (dataset['pos_u'], dataset['pos_v'])), shape=(num_nodes, num_nodes)).tocsr()
    features = sp.csr_matrix(dataset['cora_features'])
    return adj, features


def pseudo_load_data_with_features(path_to_dataset: str):
    num_nodes = 1000
    u, v = np.random.randint(0, num_nodes, size=(1000,)), np.random.randint(0, num_nodes, size=(1000,))
    adj = sp.coo_matrix((np.ones(shape=(len(u),)), (u, v)), shape=(num_nodes, num_nodes)).tocsr()
    features = sp.csr_matrix(np.random.randn(num_nodes, 10))
    return adj, features


load_data_with_features = real_load_data_with_features


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    '''
    进行评估
    :param emb:经过图卷积的embedding
    :param adj_orig:除去对角元素的邻接矩阵
    :param edges_pos:正样本，有链接关系
    :param edges_neg:负样本，无链接关系
    :return:
    '''

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # predict on val set and test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


class Train():
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        pass

    def train_model(self):
        # load config file

        # data catalog path
        # data_catalog = config.get(section, "data_catalog")

        # node cites path
        # node_cites_path = config.get(section, "node_cites_path")
        # node_cites_path = os.path.join(data_catalog, node_cites_path)

        # node features path
        # node_features_path = config.get(section, 'node_features_path')
        # node_features_path = os.path.join(data_catalog, node_features_path)

        # model save/load path
        # model_path = config.get(section, "model_path")

        # model param config

        model_path = self.cfg.model_path
        hidden_dim1 = self.cfg.hidden_dim1
        hidden_dim2 = self.cfg.hidden_dim2
        hidden_dim3 = self.cfg.hidden_dim3
        dropout = self.cfg.dropout
        vae_bool = self.cfg.vae_bool
        alpha = self.cfg.alpha
        lr = self.cfg.lr
        lr_decay = self.cfg.lr_decay
        weight_decay = self.cfg.weight_decay
        clip = self.cfg.clip
        epochs = self.cfg.epochs

        # if with_feats:
        #     # 加载带节点特征的数据集
        #     adj, features = load_data_with_features(node_cites_path, node_features_path)
        # else:
        #     # 加载不带节点特征的数据集
        #     adj = load_data_without_features(node_cites_path)
        #     features = sp.identity(adj.shape[0])
        adj, features = load_data_with_features(to_absolute_path(self.cfg.train_dataset_path))

        num_nodes = adj.shape[0]
        num_edges = adj.sum()

        features = sparse_to_tuple(features)
        num_features = features[2][1]

        # 去除对角线元素
        # 下边的右部分为：返回adj_orig的对角元素（一维），并增加一维，抽出adj_orig的对角元素并构建只有这些对角元素的对角矩阵
        adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj_orig.eliminate_zeros()

        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_orig)

        adj = adj_train

        # 返回D^{-0.5}SD^{-0.5}的coords, data, shape，其中S=A+I
        adj_norm = preprocess_graph(adj)
        adj_label = adj_train + sp.eye(adj_train.shape[0])
        # adj_label = sparse_to_tuple(adj_label)

        adj_label = torch.HalfTensor(adj_label.toarray()).to(self.device)
        # adj_label_coo = adj_label.tocoo()
        # adj_label = torch.sparse.FloatTensor(torch.LongTensor([adj_label_coo.row.tolist(), adj_label_coo.col.tolist()]),
        #                                      torch.LongTensor(adj_label_coo.data.astype(np.int32))).to(self.device)
        '''
        注意，adj的每个元素非1即0。pos_weight是用于训练的邻接矩阵中负样本边（既不存在的边）和正样本边的倍数（即比值），这个数值在二分类交叉熵损失函数中用到，
        如果正样本边所占的比例和负样本边所占比例失衡，比如正样本边很多，负样本边很少，那么在求loss的时候可以提供weight参数，将正样本边的weight设置小一点，负样本边的weight设置大一点，
        此时能够很好的平衡两类在loss中的占比，任务效果可以得到进一步提升。参考：https://www.zhihu.com/question/383567632
        负样本边的weight都为1，正样本边的weight都为pos_weight
        '''
        pos_weight = float(adj.shape[0] * adj.shape[0] - num_edges) / num_edges
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        # create model
        print('create model ...')
        model = GATModelVAE(num_features, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, hidden_dim3=hidden_dim3, dropout=dropout, alpha=alpha, vae_bool=vae_bool)

        # define optimizer

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        model = model.to(self.device)
        # 稀疏张量被表示为一对致密张量：一维张量和二维张量的索引。可以通过提供这两个张量来构造稀疏张量
        adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                            torch.FloatTensor(adj_norm[1]),
                                            torch.Size(adj_norm[2]))
        features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                            torch.FloatTensor(features[1]),
                                            torch.Size(features[2])).to_dense()
        adj_norm = adj_norm.to(self.device)
        features = features.to(self.device)
        norm = torch.FloatTensor(np.array(norm)).to(self.device)
        pos_weight = torch.tensor(pos_weight).to(self.device)
        num_nodes = torch.tensor(num_nodes).to(self.device)

        print('start training...')
        best_valid_roc_score = float('-inf')
        hidden_emb = None
        model.train()
        for epoch in range(epochs):
            t = time.time()
            optimizer.zero_grad()
            recovered, mu, logvar = model(features, adj_norm)
            if vae_bool:
                loss = vgae_loss_function(preds=recovered, labels=adj_label,
                                          mu=mu, logvar=logvar, n_nodes=num_nodes,
                                          norm=norm, pos_weight=pos_weight)
            else:
                loss = gae_loss_function(preds=recovered, labels=adj_label, norm=norm, pos_weight=pos_weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            cur_loss = loss.item()
            optimizer.step()

            hidden_emb = mu.data.cpu().numpy()
            # 评估验证集，val set
            roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
            # 保存最好的roc score
            if roc_score > best_valid_roc_score:
                best_valid_roc_score = roc_score
                # 不需要保存整个model，只需保存hidden_emb，因为后面的解码是用hidden_emb内积的形式作推断
                with open(os.path.join(model_path, f'epoch={epoch}.roc_score={best_valid_roc_score}.pkl'), 'wb') as f:
                    pickle.dump(hidden_emb, f)

            print("Epoch:", '%04d' % (epoch + 1), "train_loss = ", "{:.5f}".format(cur_loss),
                  "val_roc_score = ", "{:.5f}".format(roc_score),
                  "average_precision_score = ", "{:.5f}".format(ap_score),
                  "time=", "{:.5f}".format(time.time() - t)
                  )

        print("Optimization Finished!")

        # 评估测试集，test set
        roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
        print('test roc score: {}'.format(roc_score))
        print('test ap score: {}'.format(ap_score))


@hydra.main(config_path=".", config_name="config")
def main(cfg):
    DEVICE = torch.device('cuda:2')
    train = Train(cfg, DEVICE)
    train.train_model()


if __name__ == '__main__':
    main()
