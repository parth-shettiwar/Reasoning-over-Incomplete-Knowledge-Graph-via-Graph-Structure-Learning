import torch as th
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import dgl
import dgl.function as fn

from utils import ccorr
import math

class CompGraphConv(nn.Module):
    """One layer of CompGCN."""

    def __init__(self,
                 in_dim,
                 out_dim,
                 num_relations,
                 num_bases,
                 comp_fn='sub',
                 batchnorm=True,
                 dropout=0.1):
        super(CompGraphConv, self).__init__()
        self.n_heads = 5
        self.num_bases = num_bases
        self.d_k  = out_dim // self.n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.comp_fn = comp_fn
        self.actvation = th.tanh
        self.batchnorm = batchnorm
        self.num_rels = num_relations

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)

        self.W_S = nn.Linear(self.in_dim, self.out_dim)

        self.k_linear = nn.Linear(self.in_dim, self.out_dim)
        self.q_linear = nn.Linear(self.in_dim, self.out_dim)
        self.v_linear = nn.Linear(self.in_dim, self.out_dim)


        # self.relation_att = nn.Parameter(th.Tensor(self.num_bases, self.out_dim))

        self.relation_att = nn.Parameter(th.Tensor(self.num_bases, self.out_dim))

        nn.init.xavier_normal_(self.relation_att)
        
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(th.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_normal_(self.w_comp)

        # self.relation_pri   = nn.Parameter(th.ones(num_relations, self.n_heads))
        # self.relation_att   = nn.Parameter(th.Tensor(num_relations, self.n_heads, self.d_k, self.d_k))
        # self.relation_msg   = nn.Parameter(th.Tensor(num_relations, self.n_heads, self.d_k, self.d_k))

        # self.fc = nn.Linear(self.in_dim, self.out_dim)
        # self.att_fc = nn.Linear(self.out_dim, 1)
        
        # define relation transform layer
        self.W_R = nn.Linear(self.in_dim, self.out_dim)
        # self.W_R_att = nn.Linear(self.in_dim, self.out_dim)

        self.alpha = nn.Parameter(th.ones(1))

        #self loop embedding
        self.loop_rel = nn.Parameter(th.Tensor(1, self.in_dim))
        nn.init.xavier_normal_(self.loop_rel)

    def edge_attention(self, edges):

        src_k = self.k_linear(edges.src['h']) #544230, 200
        dst_q = self.q_linear(edges.dst['h'])
        src_v = self.v_linear(edges.src['h'])

        return {'v': src_v, 'k': src_k, 'q': dst_q}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)

        w = self.relations_weights[edges.data['etype']]
        att = (edges.data['k'] * w * edges.data['q']).sum(-1)
        # att = (edges.data['k'] * self.W_R_att(edges.data['h']) * edges.data['q']).sum(-1)
        
        return {'v': edges.data['v'], 'att': att}


    def reduce_func(self, nodes):
        att_weights = F.softmax(nodes.mailbox['att'], dim=1).unsqueeze(1)
        h = th.sum(th.bmm(att_weights, nodes.mailbox['v']), dim=1)
        return {'final': h}

    def forward(self, g, n_in_feats, r_feats):

        with g.local_scope():

            g.srcdata['h'] = n_in_feats
            r_feats = th.cat((r_feats, self.loop_rel), 0)

            g.edata['h'] = r_feats[g.edata['etype']] * g.edata['norm']

            coefficient = self.w_comp  # 50
            self.relations_weights = th.matmul(coefficient, self.relation_att.view(self.num_bases, -1))

            g.apply_edges(func=self.edge_attention)

            g.update_all(self.message_func, self.reduce_func)

            # Sum all of the comp results as output of nodes and dropout
            alpha = th.sigmoid(self.alpha[0])

            n_out_feats = alpha*self.W_S(g.ndata['h']) + (1-alpha)*self.dropout(g.ndata['final'])

            # Compute relation output
            r_out_feats = self.W_R(r_feats)

            # Batch norm
            if self.batchnorm:
                n_out_feats = self.bn(n_out_feats)

            # Activation function
            if self.actvation is not None:
                n_out_feats = self.actvation(n_out_feats)

        return n_out_feats, r_out_feats[:-1]


class CompGCN(nn.Module):
    def __init__(self,
                 num_bases,
                 num_rel,
                 num_ent,
                 in_dim=100,
                 layer_size=[200],
                 comp_fn='sub',
                 batchnorm=True,
                 dropout=0.1,
                 layer_dropout=[0.3],
                 emb_ent=None,
                 emb_relations=None):
        super(CompGCN, self).__init__()

        self.num_bases = num_bases
        self.num_rel = num_rel
        self.num_ent = num_ent
        self.in_dim = in_dim
        self.layer_size = layer_size
        self.comp_fn = comp_fn
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.layer_dropout = layer_dropout
        self.num_layer = len(layer_size)

       #CompGCN layers
        self.layers = nn.ModuleList()
        self.layers.append(
            CompGraphConv(self.in_dim, self.layer_size[0], num_rel, num_bases, comp_fn = self.comp_fn, batchnorm=self.batchnorm, dropout=self.dropout)
        )
        for i in range(self.num_layer-1):
            self.layers.append(
                CompGraphConv(self.layer_size[i], self.layer_size[i+1], num_rel, num_bases, comp_fn = self.comp_fn, batchnorm=self.batchnorm, dropout=self.dropout)
            )

        #Initial relation embeddings
        
        if self.num_bases > 0:
            self.basis = nn.Parameter(th.Tensor(self.num_bases, self.in_dim))
            self.weights = nn.Parameter(th.Tensor(self.num_rel, self.num_bases))
            nn.init.xavier_normal_(self.basis)
            nn.init.xavier_normal_(self.weights)
        
        else:
            self.rel_embds = nn.Parameter(th.Tensor(self.num_rel, self.in_dim))
            nn.init.xavier_normal_(self.rel_embds)

        self.n_embds = nn.Parameter(th.Tensor(self.num_ent, self.in_dim))
        nn.init.xavier_normal_(self.n_embds)
        
        #Dropout after compGCN layers
        self.dropouts = nn.ModuleList()
        for i in range(self.num_layer):
            self.dropouts.append(
                nn.Dropout(self.layer_dropout[i])
            )


    def forward(self, graph):
        #node and relation features
        n_feats = self.n_embds
        if self.num_bases > 0:
            r_embds = th.mm(self.weights, self.basis)
            r_feats = r_embds
        else:
            r_feats = self.rel_embds

        for layer, dropout in zip(self.layers, self.dropouts):
            n_feats, r_feats = layer(graph, n_feats, r_feats)
            n_feats = dropout(n_feats)

        return n_feats, r_feats

#Use convE as the score function
class CompGCN_ConvE(nn.Module):
    def __init__(self,
                 num_bases,
                 num_rel,
                 num_ent,
                 in_dim,
                 layer_size,
                 comp_fn='sub',
                 batchnorm=True,
                 dropout=0.1,
                 layer_dropout=[0.3],
                 num_filt=200,
                 hid_drop=0.3,
                 feat_drop=0.3,
                 ker_sz=5,
                 k_w=5,
                 k_h=5,
                 emb=None,
                 rel=None
                 ):
        super(CompGCN_ConvE, self).__init__()

        self.embed_dim = layer_size[-1]
        self.hid_drop=hid_drop
        self.feat_drop=feat_drop
        self.ker_sz=ker_sz
        self.k_w=k_w
        self.k_h=k_h
        self.num_filt=num_filt

        print(emb)

        #compGCN model to get sub/rel embs
        self.compGCN_Model = CompGCN(num_bases, num_rel, num_ent, in_dim, layer_size, comp_fn, batchnorm, dropout, layer_dropout, emb_ent=emb, emb_relations=rel)
        
        #batchnorms to the combined (sub+rel) emb
        self.bn0 = th.nn.BatchNorm2d(1)
        self.bn1 = th.nn.BatchNorm2d(self.num_filt)
        self.bn2 = th.nn.BatchNorm1d(self.embed_dim)
        
        #dropouts and conv module to the combined (sub+rel) emb
        self.hidden_drop = th.nn.Dropout(self.hid_drop)
        self.feature_drop = th.nn.Dropout(self.feat_drop)
        self.m_conv1 = th.nn.Conv2d(1, out_channels=self.num_filt, kernel_size=(self.ker_sz, self.ker_sz), stride=1, padding=0, bias=False)
        
        flat_sz_h = int(2 * self.k_w) - self.ker_sz + 1
        flat_sz_w = self.k_h - self.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
        self.fc = th.nn.Linear(self.flat_sz, self.embed_dim) 

        #bias to the score
        self.bias = nn.Parameter(th.zeros(num_ent))
        
    #combine entity embeddings and relation embeddings
    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_inp = th.cat([e1_embed, rel_embed], 1)
        stack_inp = th.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.k_w, self.k_h))
        return stack_inp
    
    def forward(self, graph, sub, rel):
        #get sub_emb and rel_emb via compGCN
        n_feats, r_feats = self.compGCN_Model(graph)
        sub_emb = n_feats[sub, :]
        rel_emb = r_feats[rel, :]

        #combine the sub_emb and rel_emb
        stk_inp = self.concat(sub_emb, rel_emb)
        #use convE to score the combined emb
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        #compute score
        x = th.mm(x, n_feats.transpose(1,0))
        #add in bias
        x += self.bias.expand_as(x)
        score = th.sigmoid(x)
        return score
        
