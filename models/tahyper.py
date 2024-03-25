import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hgcn.manifolds as manifolds
import hgcn.models.encoders as encoders
from hgcn.models.decoders import model2decoder
from hgcn.layers.layers import GraphConvolution, Linear


class TAHyper(nn.Module):
    """
    Base model for graph embedding tasks.
    """
    def __init__(self, args):
        super(TAHyper, self).__init__()
        # self.n_class = 2
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
        print("encoder:", args.model)

        nhid = args.dim
        self.n_out = args.nout

        hnn_layers_y1 = []
        hnn_layers_y0 = []
            
            
        for i in range(self.n_out):
            hnn_layers_y1 += [Linear(nhid, nhid, args.dropout, F.relu, args.bias),]
            hnn_layers_y0 += [Linear(nhid, nhid, args.dropout, F.relu, args.bias),]

        self.layers_y1 = nn.Sequential(*hnn_layers_y1)
        self.layers_y0 = nn.Sequential(*hnn_layers_y0)

        # self.y1_out = hyp_layers.HypLinear(self.manifold, nhid, 1, self.c, args.dropout, use_bias=True)
        # self.y0_out = hyp_layers.HypLinear(self.manifold, nhid, 1, self.c, args.dropout, use_bias=True)

        self.y1_out = Linear(nhid, 1, args.dropout, lambda x: x, args.bias)
        self.y0_out = Linear(nhid, 1, args.dropout, lambda x: x, args.bias)

        # self.dc = FermiDiracDecoder(r=args.r, t=args.t)

        self.out_nn1 = Linear(nhid*2, nhid, args.dropout, F.relu, args.bias)
        self.out_nn = Linear(nhid, 1, args.dropout, F.sigmoid, args.bias)

        # self.out_nn = Linear(nhid*2, 1, args.dropout, F.sigmoid, args.bias)


    def encode(self, x, adj):

        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h
    

    def uplift_decode(self, rep, t):

        # euclidean
        rep = self.manifold.proj_tan0(self.manifold.logmap0(rep, c=self.c), c=self.c)
        # y1 = self.manifold.proj_tan0(self.manifold.logmap0(rep, c=self.c), c=self.c)
        y0 = self.layers_y0(rep)
        y1 = self.layers_y1(rep)

        y0 = self.y0_out(y0).view(-1)
        y1 = self.y1_out(y1).view(-1)

        y = torch.where(t > 0,y1,y0)

        p1 = 0.0
        return y, p1
    
    
    def lp_cls(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        
        h = self.manifold.proj_tan0(self.manifold.logmap0(h, c=self.c), c=self.c)
        # print("idx", idx.shape)
        emb_in = h[idx[:, :, 0], :]
        emb_out_samet = h[idx[:, :, 1], :]
        emb_out_difft = h[idx[:, :, 2], :]
        # print("emb_in", emb_in.shape)

        pos_input = torch.cat([emb_in, emb_out_samet], axis=-1)
        neg_input = torch.cat([emb_in, emb_out_difft], axis=-1)
        pos_input = self.out_nn1(pos_input)
        neg_input = self.out_nn1(neg_input)
        pos_scores = self.out_nn(pos_input).squeeze()
        neg_scores = self.out_nn(neg_input).squeeze()
        # print("pos_scores", pos_scores)

        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores), reduction="none")
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores), reduction="none")
        # print("loss", loss.shape)
        loss = loss.mean(axis=-1)
 
        return loss

    
    def forward(self, x, adj, t, idx):

        rep = self.encode(x, adj)
        y, p1 = self.uplift_decode(rep, t)
        # print("finish./...")
        minus = self.lp_cls(rep, idx)
        
        return y, rep, p1, minus
    



# class HGCN_DECONF_RES(nn.Module):
#     """
#     Base model for graph embedding tasks.
#     """

#     def __init__(self, args):
#         super(HGCN_DECONF_RES, self).__init__()
#         # self.n_class = 2
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

#         self.y1_out = Linear(nhid, 1, args.dropout, lambda x: x, args.bias)
#         self.y0_out = Linear(nhid, 1, args.dropout, lambda x: x, args.bias)

#         self.res_nn = Linear(args.feat_dim, nhid, args.dropout, F.relu, args.bias)


#     def encode(self, x, adj):

#         if self.manifold.name == 'Hyperboloid':
#             o = torch.zeros_like(x)
#             x = torch.cat([o[:, 0:1], x], dim=1)
#         h = self.encoder.encode(x, adj)
#         return h
    

#     def uplift_decode(self, rep, t):
#         # # hyperbolic
#         # y0 = self.layers_y0(rep)
#         # y1 = self.layers_y1(rep)
#         # y0 = self.manifold.proj_tan0(self.manifold.logmap0(y0, c=self.c), c=self.c)
#         # y1 = self.manifold.proj_tan0(self.manifold.logmap0(y1, c=self.c), c=self.c)
#         # euclidean
        

#         y0 = self.layers_y0(rep)
#         y1 = self.layers_y1(rep)

#         y0 = self.y0_out(y0).view(-1)
#         y1 = self.y1_out(y1).view(-1)

#         y = torch.where(t > 0,y1,y0)

#         p1 = 0.0
#         return y, p1
    
    
#     def forward(self, x, adj, t):

#         rep = self.encode(x, adj)
#         rep = self.manifold.proj_tan0(self.manifold.logmap0(rep, c=self.c), c=self.c)
#         rep2 = self.res_nn(x)
#         rep += rep2

#         y, p1 = self.uplift_decode(rep, t)
        
#         return y, rep, p1