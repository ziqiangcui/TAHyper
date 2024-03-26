"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from hgcn.layers.layers import FermiDiracDecoder
import hgcn.layers.hyp_layers as hyp_layers
import hgcn.manifolds as manifolds
import hgcn.models.encoders as encoders
from hgcn.models.decoders import model2decoder
from hgcn.utils.eval_utils import acc_f1
from hgcn.layers.layers import GraphConvolution, Linear


# class iDAUM_orthogonal(nn.Module):
#     """
#     Base model for graph embedding tasks.
#     """
#     def __init__(self, args):
#         super(iDAUM_orthogonal, self).__init__()
#         self.n_class = 2
#         self.manifold_name = args.manifold
#         if args.c is not None:
#             self.c = torch.tensor([args.c])
#             if not args.cuda == -1:
#                 self.c = self.c.to(args.device)
#         else:
#             self.c = nn.Parameter(torch.Tensor([1.]))
#         self.manifold = getattr(manifolds, self.manifold_name)()
#         if self.manifold.name == 'Hyperboloid':
#             args.feat_dim = args.feat_dim + 1
            
#         self.nnodes = args.n_nodes
#         self.encoder = getattr(encoders, args.model)(self.c, args)
#         print("encoder:", args.model)

#         nhid = args.dim
#         self.n_out = args.nout

#         hnn_layers_y1 = []
#         hnn_layers_y0 = []
#         # for i in range(self.n_out):
#         #     hnn_layers_y1 += [hyp_layers.HypLinear(self.manifold, nhid, nhid, self.c, args.dropout, use_bias=True),
#         #                          hyp_layers.HypAct(self.manifold, self.c, self.c, F.relu)]
#         #     hnn_layers_y0 += [hyp_layers.HypLinear(self.manifold, nhid, nhid, self.c, args.dropout, use_bias=True),
#         #                          hyp_layers.HypAct(self.manifold, self.c, self.c, F.relu)]
        
#         for i in range(self.n_out):
#             hnn_layers_y1 += [Linear(nhid, nhid, args.dropout, F.relu, args.bias),]
#             hnn_layers_y0 += [Linear(nhid, nhid, args.dropout, F.relu, args.bias),]

#         self.layers_y1 = nn.Sequential(*hnn_layers_y1)
#         self.layers_y0 = nn.Sequential(*hnn_layers_y0)

#         self.auxtask_nn = nn.Sequential(
#             hyp_layers.HypLinear(self.manifold, nhid, nhid, self.c, args.dropout, use_bias=True),
#             hyp_layers.HypAct(self.manifold, self.c, self.c, F.relu),
#             # hyp_layers.HypLinear(self.manifold, nhid, 1, args.c, args.dropout, use_bias=True),
#             # hyp_layers.HypAct(self.manifold, args.c, args.c, F.sigmoid),
#         )

#         self.aux_rep_nn = Linear(nhid, nhid, args.dropout, lambda x: x, 0)
#         self.uplift_rep_nn = Linear(nhid, nhid, args.dropout, lambda x: x, 0)
#         self.aux_cls = Linear(nhid, self.n_class, args.dropout, lambda x: x, args.bias)
#         self.y1_out = Linear(nhid, 1, args.dropout, lambda x: x, args.bias)
#         self.y0_out = Linear(nhid, 1, args.dropout, lambda x: x, args.bias)


#     def encode(self, x, adj, seed_indicator):

#         # print("debug", self.layers_y1[0].linear.weight.shape) 
#         if self.manifold.name == 'Hyperboloid':
#             o = torch.zeros_like(x)
#             x = torch.cat([o[:, 0:1], x], dim=1)
#         h = self.encoder.encode(x, adj)
#         return h
    
#     def diffusion_decode(self, rep):
#         # rep = self.auxtask_nn(rep)
#         rep = self.manifold.proj_tan0(self.manifold.logmap0(rep, c=self.c), c=self.c)
#         # y_diff = F.sigmoid(self.aux_cls(rep))
#         rep = self.aux_rep_nn(rep)
#         y_diff = self.aux_cls(rep)
#         y_diff = F.log_softmax(y_diff, dim=1)
#         # print("y_diff", y_diff.shape)
#         return y_diff

#     def uplift_decode(self, rep, t):

#         rep = self.manifold.proj_tan0(self.manifold.logmap0(rep, c=self.c), c=self.c)
#         rep = self.uplift_rep_nn(rep)
       
#         y0 = self.layers_y0(rep)
#         y1 = self.layers_y1(rep)
#         y0 = self.y0_out(y0).view(-1)
#         y1 = self.y1_out(y1).view(-1)

#         y = torch.where(t > 0,y1,y0)

#         # print("y", y.shape)

#         p1 = 0.0
#         return y, p1
    
    
#     def forward(self, x, adj, t, seed_indicator):

#         rep = self.encode(x, adj, seed_indicator)
#         y_diff = self.diffusion_decode(rep)
#         y, p1 = self.uplift_decode(rep, t)

#         w_uplift = self.uplift_rep_nn.linear.weight.sum(axis=1, keepdim=True)
#         w_aux = self.aux_rep_nn.linear.weight.sum(axis=1, keepdim=True)
        
#         # print("w_uplift", w_uplift) 
#         # print("w_aux", w_aux)
#         orthogonal = torch.matmul(w_uplift.T, w_aux)
#         orthogonal = orthogonal**2
#         return y, y_diff, rep, p1, orthogonal

# class iDAUM(nn.Module):
#     """
#     Base model for graph embedding tasks.
#     """

#     def __init__(self, args):
#         super(iDAUM, self).__init__()
#         self.n_class = 2
#         self.manifold_name = args.manifold
#         if args.c is not None:
#             self.c = torch.tensor([args.c])
#             if not args.cuda == -1:
#                 self.c = self.c.to(args.device)
#         else:
#             self.c = nn.Parameter(torch.Tensor([1.]))
#         self.manifold = getattr(manifolds, self.manifold_name)()
#         if self.manifold.name == 'Hyperboloid':
#             args.feat_dim = args.feat_dim + 1
            
#         self.nnodes = args.n_nodes
#         self.encoder = getattr(encoders, args.model)(self.c, args)
#         print("encoder:", args.model)

#         nhid = args.dim
#         self.n_out = args.nout

#         hnn_layers_y1 = []
#         hnn_layers_y0 = []
#         for i in range(self.n_out):
#             hnn_layers_y1 += [hyp_layers.HypLinear(self.manifold, nhid, nhid, self.c, args.dropout, use_bias=True),
#                                  hyp_layers.HypAct(self.manifold, self.c, self.c, F.relu)]
#             hnn_layers_y0 += [hyp_layers.HypLinear(self.manifold, nhid, nhid, self.c, args.dropout, use_bias=True),
#                                  hyp_layers.HypAct(self.manifold, self.c, self.c, F.relu)]

#         self.layers_y1 = nn.Sequential(*hnn_layers_y1)
#         self.layers_y0 = nn.Sequential(*hnn_layers_y0)

#         self.auxtask_nn = nn.Sequential(
#             hyp_layers.HypLinear(self.manifold, nhid, nhid, self.c, args.dropout, use_bias=True),
#             hyp_layers.HypAct(self.manifold, self.c, self.c, F.relu),
#             # hyp_layers.HypLinear(self.manifold, nhid, 1, args.c, args.dropout, use_bias=True),
#             # hyp_layers.HypAct(self.manifold, args.c, args.c, F.sigmoid),
#         )
#         self.aux_cls = Linear(nhid, self.n_class, args.dropout, lambda x: x, args.bias)
#         self.y1_out = Linear(nhid, 1, args.dropout, lambda x: x, args.bias)
#         self.y0_out = Linear(nhid, 1, args.dropout, lambda x: x, args.bias)
#         # self.seed_emb = nn.Embedding(2, 1)
#         # self.feat_map = nn.Linear(2280, nhid-1)

#     def encode(self, x, adj, seed_indicator):

#         # x = self.feat_map(x)
#         # ind_embedding = self.seed_emb(seed_indicator)
#         # x = torch.cat([x, ind_embedding], axis=1)

#         if self.manifold.name == 'Hyperboloid':
#             o = torch.zeros_like(x)
#             x = torch.cat([o[:, 0:1], x], dim=1)
#         h = self.encoder.encode(x, adj)
#         return h
    
#     def diffusion_decode(self, rep):
#         # rep = self.auxtask_nn(rep)
#         rep = self.manifold.proj_tan0(self.manifold.logmap0(rep, c=self.c), c=self.c)
#         # y_diff = F.sigmoid(self.aux_cls(rep))
#         y_diff = self.aux_cls(rep)
#         y_diff = F.log_softmax(y_diff, dim=1)
#         # print("y_diff", y_diff.shape)
#         return y_diff

#     def uplift_decode(self, rep, t):

#         y0 = self.layers_y0(rep)
#         y1 = self.layers_y1(rep)

#         ## no hnn
#         # y0 = rep
#         # y1 = rep
#         y0 = self.manifold.proj_tan0(self.manifold.logmap0(y0, c=self.c), c=self.c)
#         y1 = self.manifold.proj_tan0(self.manifold.logmap0(y1, c=self.c), c=self.c)
#         y0 = self.y0_out(y0).view(-1)
#         y1 = self.y1_out(y1).view(-1)

#         y = torch.where(t > 0,y1,y0)

#         # print("y", y.shape)

#         p1 = 0.0
#         return y, p1
    
    
#     def forward(self, x, adj, t, seed_indicator):

#         rep = self.encode(x, adj, seed_indicator)
#         y_diff = self.diffusion_decode(rep)
#         y, p1 = self.uplift_decode(rep, t)
        
#         return y, y_diff, rep, p1


# class BaseModelUplift(nn.Module):
#     """
#     Base model for graph embedding tasks.
#     """

#     def __init__(self, args):
#         super(BaseModelUplift, self).__init__()
#         self.n_class = 2
#         self.manifold_name = args.manifold
#         if args.c is not None:
#             self.c = torch.tensor([args.c])
#             if not args.cuda == -1:
#                 self.c = self.c.to(args.device)
#         else:
#             self.c = nn.Parameter(torch.Tensor([1.]))
#         self.manifold = getattr(manifolds, self.manifold_name)()
#         if self.manifold.name == 'Hyperboloid':
#             args.feat_dim = args.feat_dim + 1
#         self.nnodes = args.n_nodes
#         self.encoder = getattr(encoders, args.model)(self.c, args)
#         print("encoder:", args.model)

#         nhid = args.dim
#         self.n_out = args.nout

#         hnn_layers_y1 = []
#         hnn_layers_y0 = []
#         for i in range(self.n_out):
#             hnn_layers_y1 += [hyp_layers.HypLinear(self.manifold, nhid, nhid, self.c, args.dropout, use_bias=True),
#                                  hyp_layers.HypAct(self.manifold, self.c, self.c, F.relu)]
#             hnn_layers_y0 += [hyp_layers.HypLinear(self.manifold, nhid, nhid, self.c, args.dropout, use_bias=True),
#                                  hyp_layers.HypAct(self.manifold, self.c, self.c, F.relu)]
            
#         # for i in range(self.n_out):
#         #     hnn_layers_y1 += [Linear(nhid, nhid, args.dropout, F.relu, args.bias),]
#         #     hnn_layers_y0 += [Linear(nhid, nhid, args.dropout, F.relu, args.bias),]

#         self.layers_y1 = nn.Sequential(*hnn_layers_y1)
#         self.layers_y0 = nn.Sequential(*hnn_layers_y0)

#         self.auxtask_nn = nn.Sequential(
#             hyp_layers.HypLinear(self.manifold, nhid, nhid, self.c, args.dropout, use_bias=True),
#             hyp_layers.HypAct(self.manifold, self.c, self.c, F.relu),
#             # hyp_layers.HypLinear(self.manifold, nhid, 1, args.c, args.dropout, use_bias=True),
#             # hyp_layers.HypAct(self.manifold, args.c, args.c, F.sigmoid),
#         )
#         self.aux_cls = Linear(nhid, self.n_class, args.dropout, lambda x: x, args.bias)
#         self.y1_out = Linear(nhid, 1, args.dropout, lambda x: x, args.bias)
#         self.y0_out = Linear(nhid, 1, args.dropout, lambda x: x, args.bias)


#     def encode(self, x, adj):

#         if self.manifold.name == 'Hyperboloid':
#             o = torch.zeros_like(x)
#             x = torch.cat([o[:, 0:1], x], dim=1)
#         h = self.encoder.encode(x, adj)
#         return h
    
#     def diffusion_decode(self, rep):
#         rep = self.auxtask_nn(rep)
#         rep = self.manifold.proj_tan0(self.manifold.logmap0(rep, c=self.c), c=self.c)
#         y_diff = F.sigmoid(self.aux_cls(rep))
#         # y_diff = self.aux_cls(rep)
#         # y_diff = F.log_softmax(y_diff, dim=1)
#         # print("y_diff", y_diff.shape)
#         return y_diff

#     def uplift_decode(self, rep, t):
#         # hyperbolic
#         y0 = self.layers_y0(rep)
#         y1 = self.layers_y1(rep)
#         y0 = self.manifold.proj_tan0(self.manifold.logmap0(y0, c=self.c), c=self.c)
#         y1 = self.manifold.proj_tan0(self.manifold.logmap0(y1, c=self.c), c=self.c)
#         # # euclidean
#         # y0 = self.manifold.proj_tan0(self.manifold.logmap0(rep, c=self.c), c=self.c)
#         # y1 = self.manifold.proj_tan0(self.manifold.logmap0(rep, c=self.c), c=self.c)
#         # y0 = self.layers_y0(y0)
#         # y1 = self.layers_y1(y1)


#         y0 = self.y0_out(y0).view(-1)
#         y1 = self.y1_out(y1).view(-1)

#         y = torch.where(t > 0,y1,y0)

#         # print("y", y.shape)

#         p1 = 0.0
#         return y, p1
    
    
#     def forward(self, x, adj, t, seed_indicator):

#         rep = self.encode(x, adj)
#         # y_diff = self.diffusion_decode(rep)
#         y, p1 = self.uplift_decode(rep, t)
        
#         return y, 0.0, rep, p1



class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges

    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.dc.forward(sqdist)
        return probs

    def compute_metrics(self, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

