import time
import argparse
import numpy as np
from hgcn.utils.train_utils import add_flags_from_config
import hgcn.optimizers as optimizers
import torch
import torch.optim as optim
from models.tahyper import TAHyper
import utils
import random
import csv

config_args = {
    'training_config': {
        'nocuda': (0, 'Disables CUDA training'),
        'lr': (0.01, 'learning rate'),
        'dropout': (0.1, 'dropout rate'),
        'epochs': (200, 'maximum number of epochs to train for'),
        'weight-decay': (1e-3, 'l2 regularization strength'),
        'optimizer': ('RiemannianAdam', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'momentum': (0.999, 'momentum in optimizer'),
        'patience': (100, 'patience for early stopping'),
        'seed': (33, 'seed for training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'save': (0, '1 to save model and logs and 0 otherwise'),
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'sweep-c': (0, ''),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.8, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),
        'grad-clip': (100, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min-epochs': (100, 'do not early stop before min-epochs')
    }, 
    'model_config': {
        'model': ('HGCN', 'which encoder to use'),
        'dim': (100, 'embedding dimension'),
        'manifold': ('PoincareBall', 'which manifold to use, can be any of [Hyperboloid, PoincareBall]'),
        'c': (0.01, 'hyperbolic radius, set to None for trainable curvature'),
        'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        'num-layers': (2, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n-heads': (1, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'double-precision': ('0', 'whether to use double precision'),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'local-agg': (0, 'whether to local tangent space aggregation or not'),
        'balance_alpha': (1e-4, 'trade-off of representation balancing'),
        'nout': (2, ''),
        'nin': (2, ''),
        'normy': (1, 'normy'),
    },
    'data_config': {
        'tr': (0.6, 'training data ratio'),
        'path': ('./datasets/', 'dataset path'),
        'dataset': ('BlogCatalog', 'dataset name'),
        'extrastr': ('1', 'extra str'),
        'use-feats': (1, 'whether to use node features or not'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
        # 'split-seed': (1234, 'seed for data splits (train/test/val)'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)

args = parser.parse_args()

args.cuda = not args.nocuda and torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device
Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if args.cuda else torch.LongTensor

alpha = Tensor([args.alpha])

np.random.seed(args.seed)
torch.manual_seed(args.seed)

loss = torch.nn.MSELoss()
bce_loss = torch.nn.BCEWithLogitsLoss()

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    alpha = alpha.cuda()
    loss = loss.cuda()
    bce_loss = bce_loss.cuda()


def generate_nodes_t_same_diff(A, T):
    T = T.reshape(-1)
    same_node_dic = {}
    diff_node_dic = {}
    row_indices, col_indices = A.nonzero()
    for i in range(row_indices.shape[0]):
        row_idx = row_indices[i]
        col_idx = col_indices[i]
        if row_idx not in same_node_dic.keys():
            same_node_dic[row_idx] = []
            diff_node_dic[row_idx] = []
        if T[col_idx] == T[row_idx]:
            same_node_dic[row_idx].append(col_idx)
        else:
            diff_node_dic[row_idx].append(col_idx)

    to_delete = []
    for k in same_node_dic.keys():
        if len(same_node_dic[k]) == 0 or len(diff_node_dic[k]) == 0:
            to_delete.append(k)

    for k in to_delete:
        del same_node_dic[k]
        del diff_node_dic[k]
    return same_node_dic, diff_node_dic


def generate_nodes_t_same_diff_by_degree(A, T):
    T = T.reshape(-1)
    same_node_dic = {}
    diff_node_dic = {}
    row_indices, col_indices = A.nonzero()
    for i in range(row_indices.shape[0]):
        row_idx = row_indices[i]
        col_idx = col_indices[i]
        if row_idx not in same_node_dic.keys():
            same_node_dic[row_idx] = []
            diff_node_dic[row_idx] = []
        if T[col_idx] == T[row_idx]:
            same_node_dic[row_idx].append(col_idx)
        else:
            diff_node_dic[row_idx].append(col_idx)

    degree_dict = {}
    degrees = A.sum(axis=1).A1
    for i, degree in enumerate(degrees):
        degree_dict[i] = degree

    def sort_by_degree(lst):
        return sorted(lst, key=lambda x: degree_dict[x], reverse=True)
    
    for key in same_node_dic:
        same_node_dic[key] = sort_by_degree(same_node_dic[key])
        diff_node_dic[key] = sort_by_degree(diff_node_dic[key])

    to_delete = []
    for k in same_node_dic.keys():
        if len(same_node_dic[k]) == 0 or len(diff_node_dic[k]) == 0:
            to_delete.append(k)
    for k in to_delete:
        del same_node_dic[k]
        del diff_node_dic[k]
    return same_node_dic, diff_node_dic



def prepare(i_exp):

    # Load data and init models
    X, A, T, Y1, Y0 = utils.load_data(args.path, name=args.dataset, original_X=False, exp_id=str(i_exp), extra_str=args.extrastr)
    args.n_nodes, args.feat_dim = X.shape

    # same_node_dic, diff_node_dic = generate_nodes_t_same_diff(A, T)
    same_node_dic, diff_node_dic = generate_nodes_t_same_diff_by_degree(A, T)
    

    n = X.shape[0]
    n_train = int(n * args.tr)
    n_test = int(n * 0.2)
    # n_valid = n_test
    np.random.seed(42)
    idx = np.random.permutation(n)
    idx_train, idx_test, idx_val = idx[:n_train], idx[n_train:n_train+n_test], idx[n_train+n_test:]
 
    to_delete = []
    for k in same_node_dic.keys():
        if k not in idx_train:
            to_delete.append(k)
  
    for k in to_delete:
        del same_node_dic[k]
        del diff_node_dic[k]


    X = utils.normalize(X) #row-normalize
    # A = utils.normalize(A+sp.eye(n))

    X = X.todense()
    X = Tensor(X)

    Y1 = Tensor(np.squeeze(Y1))
    Y0 = Tensor(np.squeeze(Y0))
    T = LongTensor(np.squeeze(T))

    A = utils.sparse_mx_to_torch_sparse_tensor(A, cuda=args.cuda)


    idx_train = LongTensor(idx_train)
    idx_val = LongTensor(idx_val)
    idx_test = LongTensor(idx_test)

    # Model and optimizer
    
    model = TAHyper(args)
    model = model.cuda()

    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    

    return X, A, T, Y1, Y0, idx_train, idx_val, idx_test, model, optimizer, same_node_dic, diff_node_dic


def train(epoch, X, A, T, Y1, Y0, idx_train, idx_val, model, optimizer, sampled_links):
    t = time.time()
    model.train()
#    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.zero_grad()
    yf_pred, rep, p1, link_cls_loss = model(X, A, T, sampled_links)
    ycf_pred, _, p1, link_cls_loss = model(X, A, 1-T, sampled_links)

    link_cls_loss = link_cls_loss.mean()

    # representation balancing, you can try different distance metrics such as MMD
    rep_t1, rep_t0 = rep[idx_train][(T[idx_train] > 0).nonzero()], rep[idx_train][(T[idx_train] < 1).nonzero()]
    dist, _ = utils.wasserstein(rep_t1, rep_t0, cuda=args.cuda)

    YF = torch.where(T>0,Y1,Y0)
    # YCF = torch.where(T>0,Y0,Y1)

    if args.normy:
        # recover the normalized outcomes
        ym, ys = torch.mean(YF[idx_train]), torch.std(YF[idx_train])
        YFtr, YFva = (YF[idx_train] - ym) / ys, (YF[idx_val] - ym) / ys
    else:
        YFtr = YF[idx_train]
        YFva = YF[idx_val]
    
    lamb = 1.0
    loss_train = loss(yf_pred[idx_train], YFtr) + alpha * dist + lamb*link_cls_loss
 
    loss_train.backward()
    optimizer.step()

    if epoch % 10 == 0:
        # validation
        loss_val = loss(yf_pred[idx_val], YFva) + alpha * dist

        y1_pred, y0_pred = torch.where(T>0,yf_pred,ycf_pred), torch.where(T>0,ycf_pred,yf_pred)
        # Y1, Y0 = torch.where(T>0, YF, YCF), torch.where(T>0, YCF, YF)
        if args.normy:
            y1_pred, y0_pred = y1_pred * ys + ym, y0_pred * ys + ym


        # pehe_val = torch.sqrt(loss((y1_pred - y0_pred)[idx_val],(Y1 - Y0)[idx_val]))
        # mae_ate_val = torch.abs(
        #     torch.mean((y1_pred - y0_pred)[idx_val])-torch.mean((Y1 - Y0)[idx_val]))

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'link_cls_loss: {:.4f}'.format(link_cls_loss.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              # 'pehe_val: {:.4f}'.format(pehe_val.item()),
              # 'mae_ate_val: {:.4f}'.format(mae_ate_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

def eva(X, A, T, Y1, Y0, idx_train, idx_test, model, i_exp, sampled_links):
    model.eval()
    yf_pred, rep, p1, _ = model(X, A, T, sampled_links) # p1 can be used as propensity scores
    # yf = torch.where(T>0, Y1, Y0)
    ycf_pred, _, _ ,_= model(X, A, 1-T, sampled_links)

    YF = torch.where(T>0,Y1,Y0)
    YCF = torch.where(T>0,Y0,Y1)

    ym, ys = torch.mean(YF[idx_train]), torch.std(YF[idx_train])
    # YFtr, YFva = (YF[idx_train] - ym) / ys, (YF[idx_val] - ym) / ys

    y1_pred, y0_pred = torch.where(T>0,yf_pred,ycf_pred), torch.where(T>0,ycf_pred,yf_pred)

    if args.normy:
        y1_pred, y0_pred = y1_pred * ys + ym, y0_pred * ys + ym

    # Y1, Y0 = torch.where(T>0, YF, YCF), torch.where(T>0, YCF, YF)
    pehe_ts = torch.sqrt(loss((y1_pred - y0_pred)[idx_test],(Y1 - Y0)[idx_test]))
    mae_ate_ts = torch.abs(torch.mean((y1_pred - y0_pred)[idx_test])-torch.mean((Y1 - Y0)[idx_test]))
    print("Test set results:",
          "pehe_ts= {:.4f}".format(pehe_ts.item()),
          "mae_ate_ts= {:.4f}".format(mae_ate_ts.item()))

    of_path = './new_results/' + args.dataset + args.extrastr + '/' + str(args.tr)
    
    if args.lr != 1e-2:
        of_path += 'lr'+str(args.lr)
    if args.dropout != 0.5:
        of_path += 'do'+str(args.dropout)
    if args.epochs != 50:
        of_path += 'ep'+str(args.epochs)
    if args.weight_decay != 1e-5:
        of_path += 'lbd'+str(args.weight_decay)
    if args.nout != 1:
        of_path += 'nout'+str(args.nout)
    if args.alpha != 1e-5:
        of_path += 'alp'+str(args.alpha)
    if args.normy == 1:
        of_path += 'normy'

    of_path += '.csv'

    return pehe_ts.item(), mae_ate_ts.item()

def sample_link(same_node_dic, diff_node_dic):
    original_nodes = np.array(list(same_node_dic.keys())).reshape(-1, 1)
    sample_same_nodes = []
    sample_diff_nodes = []
    for k in same_node_dic.keys():
        sample_same_nodes.append(random.choice(same_node_dic[k]))
        sample_diff_nodes.append(random.choice(diff_node_dic[k]))
    sample_same_nodes = np.array(sample_same_nodes).reshape(-1, 1)
    sample_diff_nodes = np.array(sample_diff_nodes).reshape(-1, 1)
    
    original_nodes = LongTensor(original_nodes)
    sample_same_nodes = LongTensor(sample_same_nodes)
    sample_diff_nodes = LongTensor(sample_diff_nodes)
    sampled_links = torch.cat([original_nodes, sample_same_nodes, sample_diff_nodes], axis=-1)
    # print("sampled_links", sample_same_nodes)
    return sampled_links


def sample_link_multi(same_node_dic, diff_node_dic, n):
    for k in same_node_dic.keys():
        if len(same_node_dic[k]) < n:
            num_to_fill = n - len(same_node_dic[k])
            fill_list = [same_node_dic[k][0]] * num_to_fill
            same_node_dic[k].extend(fill_list)
        if len(diff_node_dic[k]) < n:
            num_to_fill = n - len(diff_node_dic[k])
            fill_list = [diff_node_dic[k][0]] * num_to_fill
            diff_node_dic[k].extend(fill_list)


    original_nodes = np.array(list(same_node_dic.keys())).reshape(-1, 1)
    sample_same_nodes = []
    sample_diff_nodes = []
    for k in same_node_dic.keys():
        sample_same_nodes.append (random.sample(same_node_dic[k], n))
        sample_diff_nodes.append (random.sample(diff_node_dic[k], n))
    sample_same_nodes = np.array(sample_same_nodes)
    sample_diff_nodes = np.array(sample_diff_nodes)
    
    original_nodes = original_nodes.repeat(n,axis=1)
    original_nodes = LongTensor(original_nodes).unsqueeze(-1)
    sample_same_nodes = LongTensor(sample_same_nodes).unsqueeze(-1)
    sample_diff_nodes = LongTensor(sample_diff_nodes).unsqueeze(-1)
    sampled_links = torch.cat([original_nodes, sample_same_nodes, sample_diff_nodes], axis=-1)

    return sampled_links


def sample_link_by_degree(same_node_dic, diff_node_dic, n):
    for k in same_node_dic.keys():
        if len(same_node_dic[k]) < n:
            num_to_fill = n - len(same_node_dic[k])
            fill_list = [same_node_dic[k][-1]] * num_to_fill
            same_node_dic[k].extend(fill_list)
        if len(diff_node_dic[k]) < n:
            num_to_fill = n - len(diff_node_dic[k])
            fill_list = [diff_node_dic[k][-1]] * num_to_fill
            diff_node_dic[k].extend(fill_list)


    original_nodes = np.array(list(same_node_dic.keys())).reshape(-1, 1)
    sample_same_nodes = []
    sample_diff_nodes = []
    for k in same_node_dic.keys():
        sample_same_nodes.append (same_node_dic[k][:n])
        sample_diff_nodes.append (diff_node_dic[k][:n])
    sample_same_nodes = np.array(sample_same_nodes)
    sample_diff_nodes = np.array(sample_diff_nodes)
    
    original_nodes = original_nodes.repeat(n,axis=1)
    original_nodes = LongTensor(original_nodes).unsqueeze(-1)
    sample_same_nodes = LongTensor(sample_same_nodes).unsqueeze(-1)
    sample_diff_nodes = LongTensor(sample_diff_nodes).unsqueeze(-1)

    sampled_links = torch.cat([original_nodes, sample_same_nodes, sample_diff_nodes], axis=-1)
    
    return sampled_links



if __name__ == '__main__':
    pehe = 0.0
    mae_ate=0.0
    for i_exp in range(0, 10): 
        # Train model
        X, A, T, Y1, Y0, idx_train, idx_val, idx_test, model, optimizer, same_node_dic, diff_node_dic = prepare(i_exp)
        sampled_links = sample_link_by_degree(same_node_dic, diff_node_dic, 10)
        t_total = time.time()
        for epoch in range(args.epochs):
            train(epoch, X, A, T, Y1, Y0, idx_train, idx_val, model, optimizer, sampled_links)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        pehe_ts, mae_ate_ts  = eva(X, A, T, Y1, Y0, idx_train, idx_test, model, i_exp, sampled_links)
        pehe += pehe_ts
        mae_ate += mae_ate_ts
    print("overrall: ", pehe/10, mae_ate/10)