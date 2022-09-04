import torch
import torch as th
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dgl
from collections import defaultdict as ddict
from ordered_set import OrderedSet
import copy

def get_subset_g(g, mask, num_rels, bidirected=False):
    src, dst = g.edges()
    sub_src = src[mask]
    sub_dst = dst[mask]
    sub_rel = g.edata['etype'][mask]

    if bidirected:
        sub_src, sub_dst = th.cat([sub_src, sub_dst]), th.cat([sub_dst, sub_src])
        sub_rel = th.cat([sub_rel, sub_rel + num_rels])

    sub_g = dgl.graph((sub_src, sub_dst), num_nodes=g.num_nodes())
    sub_g.edata[dgl.ETYPE] = sub_rel

    return sub_g

def preprocess(g, num_rels):
    # Get train graph
    train_g = get_subset_g(g, g.edata['train_mask'], num_rels)

    # Get test graph
    test_g = get_subset_g(g, g.edata['train_mask'], num_rels, bidirected=True)
    test_g.edata['norm'] = dgl.norm_by_dst(test_g).unsqueeze(-1)

    return train_g, test_g

class GlobalUniform_forTriples:
    def __init__(self, g, sample_size, p_edges_num):
        self.sample_size = sample_size
        self.eids = np.arange(p_edges_num)

    def sample(self, return_removed = False):
        kept_edges = th.from_numpy(np.random.choice(self.eids, int(self.sample_size)))
        
        if(return_removed == True):
            removed_edges = list(set(self.eids) - set(kept_edges.numpy()))
            return kept_edges, th.tensor(removed_edges)
        
        return kept_edges

class GlobalUniform:
    def __init__(self, g, sample_size):
        self.sample_size = sample_size
        self.eids = np.arange(g.num_edges())

    def sample(self, return_removed = True):
        kept_edges = th.from_numpy(np.random.choice(self.eids, self.sample_size))
        
        if(return_removed == True):
            removed_edges = list(set(th.tensor(self.eids)) - set(kept_edges))
            removed_edges = th.tensor(removed_edges)

            return kept_edges, removed_edges
        
        return kept_edges

class NeighborExpand:
    def __init__(self, g, sample_size):
        self.g = g
        self.nids = np.arange(self.g.num_nodes())
        self.sample_size = sample_size
    
    def sample(self):
        edges = th.zeros((self.sample_size), dtype=th.int64) # Will give 30,000 edges
        neighbor_counts = (self.g.in_degrees() + self.g.out_degrees()).numpy() # sums the neighbors for each entity
        seen_edge = np.array([False] * self.g.num_edges())
        seen_node = np.array([False] * self.g.num_nodes())

        for i in range(self.sample_size):
            if np.sum(seen_node) == 0:
                node_weights = np.ones_like(neighbor_counts)
                node_weights[np.where(neighbor_counts == 0)] = 0
            else:
                # Sample a visited node if applicable.
                # This guarantees a connected component.
                node_weights = neighbor_counts * seen_node

            node_probs = node_weights / np.sum(node_weights)
            chosen_node = np.random.choice(self.nids, p=node_probs)

            # Sample a neighbor of the sampled node
            u1, v1, eid1 = self.g.in_edges(chosen_node, form='all')
            u2, v2, eid2 = self.g.out_edges(chosen_node, form='all')
            u = th.cat([u1, u2])
            v = th.cat([v1, v2])
            eid = th.cat([eid1, eid2])

            to_pick = True
            while to_pick:
                random_id = th.randint(high=eid.shape[0], size=(1,))
                chosen_eid = eid[random_id]
                to_pick = seen_edge[chosen_eid]

            chosen_u = u[random_id]
            chosen_v = v[random_id]
            edges[i] = chosen_eid
            seen_node[chosen_u] = True
            seen_node[chosen_v] = True
            seen_edge[chosen_eid] = True

            neighbor_counts[chosen_u] -= 1
            neighbor_counts[chosen_v] -= 1

        return edges
        
            



class NeighborExpand_og:
    """Sample a connected component by neighborhood expansion"""
    def __init__(self, g, sample_size):
        self.g = g
        self.nids = np.arange(g.num_nodes())
        self.sample_size = sample_size

    def sample(self):
        edges = th.zeros((self.sample_size), dtype=th.int64) # Will give 30,000 edges
        neighbor_counts = (self.g.in_degrees() + self.g.out_degrees()).numpy() # sums the neighbors for each entity
        seen_edge = np.array([False] * self.g.num_edges())
        seen_node = np.array([False] * self.g.num_nodes())

        for i in range(self.sample_size):
            if np.sum(seen_node) == 0:
                node_weights = np.ones_like(neighbor_counts)
                node_weights[np.where(neighbor_counts == 0)] = 0
            else:
                # Sample a visited node if applicable.
                # This guarantees a connected component.
                node_weights = neighbor_counts * seen_node

            node_probs = node_weights / np.sum(node_weights)
            chosen_node = np.random.choice(self.nids, p=node_probs)

            # Sample a neighbor of the sampled node
            u1, v1, eid1 = self.g.in_edges(chosen_node, form='all')
            u2, v2, eid2 = self.g.out_edges(chosen_node, form='all')
            u = th.cat([u1, u2])
            v = th.cat([v1, v2])
            eid = th.cat([eid1, eid2])

            to_pick = True
            while to_pick:
                random_id = th.randint(high=eid.shape[0], size=(1,))
                chosen_eid = eid[random_id]
                to_pick = seen_edge[chosen_eid]

            chosen_u = u[random_id]
            chosen_v = v[random_id]
            edges[i] = chosen_eid
            seen_node[chosen_u] = True
            seen_node[chosen_v] = True
            seen_edge[chosen_eid] = True

            neighbor_counts[chosen_u] -= 1
            neighbor_counts[chosen_v] -= 1

        return edges

class NegativeSampler:
    def __init__(self, k=10):
        self.k = k

    def sample(self, pos_samples, num_nodes):
        batch_size = len(pos_samples)
        neg_batch_size = batch_size * self.k
        neg_samples = np.tile(pos_samples, (self.k, 1))

        values = np.random.randint(num_nodes, size=neg_batch_size)
        choices = np.random.uniform(size=neg_batch_size)
        subj = choices > 0.5
        obj = choices <= 0.5
        neg_samples[subj, 0] = values[subj]
        neg_samples[obj, 2] = values[obj]
        samples = np.concatenate((pos_samples, neg_samples))

        # binary labels indicating positive and negative samples
        labels = np.zeros(batch_size * (self.k + 1), dtype=np.float32)
        labels[:batch_size] = 1

        return th.from_numpy(samples), th.from_numpy(labels)


class IncGraphMaker:
    def __init__(self, iep, g, pos_sampler, lbl_smooth, num_workers, sample_size=30000, num_epochs=6000, batch_size=1024):
        # Shardul change
        # sample_size = 9
        # batch_size = 3
        self.g = g
        self.num_epochs = num_epochs
        self.sample_size = int(self.g.num_edges() * iep)
        # self.num_epochs = num_epochs
        # self.batch_size = batch_size
        self.lbl_smooth = lbl_smooth
        self.num_workers = num_workers

        if pos_sampler == 'neighbor':
            self.pos_sampler = NeighborExpand(self.g, self.sample_size)
        else:
            self.pos_sampler = GlobalUniform(self.g, self.sample_size)

        # self.neg_sampler = NegativeSampler()

    def get_inc_graph(self, has_etype=True):
        eids, removed_eids = self.pos_sampler.sample()
        src, dst = self.g.find_edges(eids.numpy())
        src, dst = src.numpy(), dst.numpy()
        rel = None
        if has_etype == True:
            rel = self.g.edata['etype'][eids].numpy()

        inc_g = dgl.graph((src, dst), num_nodes = self.g.num_nodes())

        return inc_g,rel

def makeGraphFromEids(g, eids):
        src, dst = g.find_edges(eids)
        src, dst = src.numpy(), dst.numpy()
        rel = None
        if has_etype == True:
            rel = self.g.edata['etype'][eids].numpy()

        inc_g = dgl.graph((src, dst), num_nodes = self.g.num_nodes())

        return inc_g,rel


class SubgraphIterator:
    def __init__(self, data, pos_sampler, lbl_smooth, num_workers, sample_size=30000, num_epochs=6000, batch_size=1024):
        # Shardul change
        sample_size = 9
        batch_size = 3
        self.g = data.g
        self.num_epochs = num_epochs
        self.sample_size = sample_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lbl_smooth = lbl_smooth
        self.num_workers = num_workers

        if pos_sampler == 'neighbor':
            self.pos_sampler = NeighborExpand(data, sample_size)
        else:
            self.pos_sampler = GlobalUniform(data, sample_size)

        self.neg_sampler = NegativeSampler()

    def __len__(self):
            return self.num_epochs

    def __getitem__(self, i):
        eids = self.pos_sampler.sample()
        src, dst = self.g.find_edges(eids)
        src, dst = src.numpy(), dst.numpy()
        rel = self.g.edata['etype'][eids].numpy()

        # relabel nodes to have consecutive node IDs
        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        num_nodes = len(uniq_v)
        # edges is the concatenation of src, dst with relabeled ID
        
        global_ids_dict = {}
        global_ids = []
        for e in edges: # [90, 10, 80, 90, 25, 100, 90] #[90, 10, 80, 25, 100] #[0, 1, 2, 0, ]
            if e not in global_ids_dict:
                global_ids_dict[e] = uniq_v[e]
                global_ids.append(uniq_v[e])

        
        src, dst = np.reshape(edges, (2, -1))
        relabeled_data = np.stack((src, rel, dst)).transpose()

        samples, labels = self.neg_sampler.sample(relabeled_data, num_nodes)

        # Use only half of the positive edges
        chosen_ids = np.random.choice(np.arange(self.sample_size),
                                      size=int(self.sample_size / 2),
                                      replace=False)

        src_sample = src[chosen_ids] # 2, 5, 0, 0 
        dst_sample = dst[chosen_ids] # 1, 0, 3, 9
        rel_sample = rel[chosen_ids] # 255, 179,  37,  63

        # we have taken the positive edges from here
        # read data and get mapping
        ent_set, rel_set = OrderedSet(), OrderedSet()
        for idx in range(len(src_sample)):
                ent_set.add(src_sample[idx])
                rel_set.add(rel_sample[idx])
                ent_set.add(dst_sample[idx])

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        #Shardul change
        #self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})
        self.rel2id.update({-1*rel: idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})


        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        new_idx = [0]*len(self.id2ent.keys())

        for i in range(len(new_idx)):
            new_idx[i] = global_ids[self.id2ent[i]]


        self.num_ent = len(self.ent2id)
        self.num_rel = len(self.rel2id) // 2

        self.data = ddict(list) #stores the triples
        sr2o = ddict(set) #The key of sr20 is (subject, relation), and the items are all the successors following (subject, relation)
        src=[]
        dst=[]
        rels = []
        inver_src = []
        inver_dst = []
        inver_rels = []

        split = 'train'
        for idx in range(len(src_sample)):

                # sub, rel, obj = map(str.lower, line.strip().split('\t'))
                sub_id, rel_id, obj_id = self.ent2id[src_sample[idx]], self.rel2id[rel_sample[idx]], self.ent2id[dst_sample[idx]]
                self.data[split].append((sub_id, rel_id, obj_id))

                if split == 'train': 
                    sr2o[(sub_id, rel_id)].add(obj_id)
                    sr2o[(obj_id, rel_id+self.num_rel)].add(sub_id) #append the reversed edges
                    src.append(sub_id)
                    dst.append(obj_id)
                    rels.append(rel_id)
                    inver_src.append(obj_id)
                    inver_dst.append(sub_id)
                    inver_rels.append(rel_id+self.num_rel)

        # Construct dgl subgraph
        src = src + inver_src
        dst = dst + inver_dst
        rels = rels + inver_rels
        self.subg = dgl.graph((src, dst), num_nodes=self.num_ent)
        self.subg.edata['etype'] = torch.Tensor(rels).long()

        #identify in and out edges
        in_edges_mask = [True] * (self.subg.num_edges()//2) + [False] * (self.subg.num_edges()//2)
        out_edges_mask = [False] * (self.subg.num_edges()//2) + [True] * (self.subg.num_edges()//2)
        self.subg.edata['in_edges_mask'] = torch.Tensor(in_edges_mask)
        self.subg.edata['out_edges_mask'] = torch.Tensor(out_edges_mask)

        #Prepare train/valid/test data
        self.data = dict(self.data)
        self.sr2o = {k: list(v) for k, v in sr2o.items()} #store only the train data

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()} #store all the data
        self.triples  = ddict(list)

        global_ids_set = set()
        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sample_type': 'pos'})
            global_ids_set.add(sub)

        sample_negs = samples[self.sample_size:]
        labels_neg = labels[self.sample_size:]

        
        for idx in range(len(sample_negs)):
            sn = sample_negs[idx]
            self.triples['train'].append({'triple':(sn[0], sn[1], -1), 'label': [], 'sample_type': 'neg'})
           

        # Now adding all the neg samples to this list itself

        self.triples = dict(self.triples)

        global_ids_subgraph = []
        for gi in global_ids_set:
            global_ids_subgraph.append(global_ids[gi])

        def get_subgraph_train_data_loader(split, batch_size, shuffle=True):
            return  DataLoader(
                    SubGraphTrainDataset(self.triples[split], num_nodes, self.lbl_smooth),
                    batch_size      = self.batch_size,
                    shuffle         = shuffle,
                    num_workers     = max(0, self.num_workers),
                    collate_fn      = TrainDataset.collate_fn
                )

        subgraph_data_loader = {}
        subgraph_data_loader['train'] = get_subgraph_train_data_loader('train', self.batch_size)

        return self.subg, uniq_v, num_nodes, subgraph_data_loader, new_idx

        # src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
        # rel = np.concatenate((rel, rel + self.num_rels))
        # sub_g = dgl.graph((src, dst), num_nodes=num_nodes)
        # sub_g.edata[dgl.ETYPE] = th.from_numpy(rel)
        # sub_g.edata['norm'] = dgl.norm_by_dst(sub_g).unsqueeze(-1)
        # uniq_v = th.from_numpy(uniq_v).view(-1).long()

        #return sub_g, uniq_v, samples, labels
        return 0

class SubgraphIterator_og:
    def __init__(self, g, num_rels, pos_sampler, sample_size=30000, num_epochs=6000):
        self.g = g
        self.num_rels = num_rels
        self.sample_size = sample_size
        self.num_epochs = num_epochs
        if pos_sampler == 'neighbor':
            self.pos_sampler = NeighborExpand(g, sample_size)
        else:
            self.pos_sampler = GlobalUniform(g, sample_size)
        self.neg_sampler = NegativeSampler()

    def __len__(self):
        return self.num_epochs

    def __getitem__(self, i):
        eids = self.pos_sampler.sample()
        src, dst = self.g.find_edges(eids)
        src, dst = src.numpy(), dst.numpy()
        rel = self.g.edata[dgl.ETYPE][eids].numpy()

        # relabel nodes to have consecutive node IDs
        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        num_nodes = len(uniq_v)
        # edges is the concatenation of src, dst with relabeled ID
        src, dst = np.reshape(edges, (2, -1))
        relabeled_data = np.stack((src, rel, dst)).transpose()

        samples, labels = self.neg_sampler.sample(relabeled_data, num_nodes)
         # till here we get our samples and labels

        # Use only half of the positive edges
        chosen_ids = np.random.choice(np.arange(self.sample_size),
                                      size=int(self.sample_size / 2),
                                      replace=False)
        # here we get our ids

        src = src[chosen_ids]
        dst = dst[chosen_ids]
        rel = rel[chosen_ids]
        src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
        rel = np.concatenate((rel, rel + self.num_rels))
        sub_g = dgl.graph((src, dst), num_nodes=num_nodes)
        sub_g.edata[dgl.ETYPE] = th.from_numpy(rel)
        sub_g.edata['norm'] = dgl.norm_by_dst(sub_g).unsqueeze(-1)
        uniq_v = th.from_numpy(uniq_v).view(-1).long()

        return sub_g, uniq_v, samples, labels

# Utility functions for evaluations (raw)

def perturb_and_get_raw_rank(emb, w, a, r, b, test_size, batch_size=100):
    """ Perturb one element in the triplets"""
    n_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    emb = emb.transpose(0, 1) # size D x V
    w = w.transpose(0, 1)     # size D x R
    for idx in range(n_batch):
        print("batch {} / {}".format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = (idx + 1) * batch_size
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        emb_ar = emb[:,batch_a] * w[:,batch_r] # size D x E
        emb_ar = emb_ar.unsqueeze(2)           # size D x E x 1
        emb_c = emb.unsqueeze(1)               # size D x 1 x V

        # out-prod and reduce sum
        out_prod = th.bmm(emb_ar, emb_c)          # size D x E x V
        score = th.sum(out_prod, dim=0).sigmoid() # size E x V
        target = b[batch_start: batch_end]

        _, indices = th.sort(score, dim=1, descending=True)
        indices = th.nonzero(indices == target.view(-1, 1), as_tuple=False)
        ranks.append(indices[:, 1].view(-1))
    return th.cat(ranks)

# Utility functions for evaluations (filtered)

def filter(triplets_to_filter, target_s, target_r, target_o, num_nodes, filter_o=True):
    """Get candidate heads or tails to score"""
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)

    # Add the ground truth node first
    if filter_o:
        candidate_nodes = [target_o]
    else:
        candidate_nodes = [target_s]

    for e in range(num_nodes):
        triplet = (target_s, target_r, e) if filter_o else (e, target_r, target_o)
        # Do not consider a node if it leads to a real triplet
        if triplet not in triplets_to_filter:
            candidate_nodes.append(e)
    return th.LongTensor(candidate_nodes)

def perturb_and_get_filtered_rank(emb, w, s, r, o, test_size, triplets_to_filter, filter_o=True):
    """Perturb subject or object in the triplets"""
    num_nodes = emb.shape[0]
    ranks = []
    for idx in range(test_size):
        if idx % 100 == 0:
            print("test triplet {} / {}".format(idx, test_size))
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        candidate_nodes = filter(triplets_to_filter, target_s, target_r,
                                 target_o, num_nodes, filter_o=filter_o)
        if filter_o:
            emb_s = emb[target_s]
            emb_o = emb[candidate_nodes]
        else:
            emb_s = emb[candidate_nodes]
            emb_o = emb[target_o]
        target_idx = 0
        emb_r = w[target_r]
        emb_triplet = emb_s * emb_r * emb_o
        scores = th.sigmoid(th.sum(emb_triplet, dim=1))

        _, indices = th.sort(scores, descending=True)
        rank = int((indices == target_idx).nonzero())
        ranks.append(rank)
    return th.LongTensor(ranks)

def _calc_mrr(emb, w, test_mask, triplets_to_filter, batch_size, filter=False):
    with th.no_grad():
        test_triplets = triplets_to_filter[test_mask]
        s, r, o = test_triplets[:,0], test_triplets[:,1], test_triplets[:,2]
        test_size = len(s)

        if filter:
            metric_name = 'MRR (filtered)'
            triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter.tolist()}
            ranks_s = perturb_and_get_filtered_rank(emb, w, s, r, o, test_size,
                                                    triplets_to_filter, filter_o=False)
            ranks_o = perturb_and_get_filtered_rank(emb, w, s, r, o,
                                                    test_size, triplets_to_filter)
        else:
            metric_name = 'MRR (raw)'
            ranks_s = perturb_and_get_raw_rank(emb, w, o, r, s, test_size, batch_size)
            ranks_o = perturb_and_get_raw_rank(emb, w, s, r, o, test_size, batch_size)

        ranks = th.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed
        mrr = th.mean(1.0 / ranks.float()).item()
        print("{}: {:.6f}".format(metric_name, mrr))

    return mrr

# Main evaluation function

def calc_mrr(emb, w, test_mask, triplets, batch_size=100, eval_p="filtered"):
    if eval_p == "filtered":
        mrr = _calc_mrr(emb, w, test_mask, triplets, batch_size, filter=True)
    else:
        mrr = _calc_mrr(emb, w, test_mask, triplets, batch_size)
    return mrr

class SubGraphTrainDataset(Dataset):
    """
    Training Dataset class.
    Parameters
    ----------
    triples: The triples used for training the model
    num_ent: Number of entities in the knowledge graph
    lbl_smooth: Label smoothing
    Returns
    -------
    A subgraph training Dataset class instance used by DataLoader
    """
    def __init__(self, triples, num_ent, lbl_smooth):
        self.triples = triples
        self.num_ent = num_ent
        self.lbl_smooth = lbl_smooth
        self.entities = np.arange(self.num_ent, dtype=np.int32)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = torch.LongTensor(ele['triple']), np.int32(ele['label'])
        trp_label = self.get_label(label)
        #label smoothing
        if self.lbl_smooth != 0.0:
            trp_label = (1.0 - self.lbl_smooth) * trp_label + (1.0 / self.num_ent)

        return triple, trp_label
    
    @staticmethod
    def collate_fn(data):
        triples = []
        labels = []
        for triple, label in data:
            triples.append(triple)
            labels.append(label)
        triple = torch.stack(triples, dim=0)
        trp_label = torch.stack(labels, dim=0)
        return triple, trp_label

    #for edges that exist in the graph, the entry is 1.0, otherwise the entry is 0.0
    def get_label(self, label):
        y = np.zeros([self.num_ent], dtype=np.float32)
        for e2 in label: 
            y[e2] = 1.0
        return torch.FloatTensor(y)


class TrainDataset(Dataset):
    """
    Training Dataset class.
    Parameters
    ----------
    triples: The triples used for training the model
    num_ent: Number of entities in the knowledge graph
    lbl_smooth: Label smoothing
    Returns
    -------
    A training Dataset class instance used by DataLoader
    """
    def __init__(self, triples, num_ent, lbl_smooth):
        self.triples = triples
        self.num_ent = num_ent
        self.lbl_smooth = lbl_smooth
        self.entities = np.arange(self.num_ent, dtype=np.int32)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label, sample_type = torch.LongTensor(ele['triple']), np.int32(ele['label']), torch.Tensor(ele['triple'])
        trp_label = self.get_label(label, sample_type)
        #label smoothing
        if self.lbl_smooth != 0.0:
            trp_label = (1.0 - self.lbl_smooth) * trp_label + (1.0 / self.num_ent)

        return triple, trp_label
    
    @staticmethod
    def collate_fn(data):
        triples = []
        labels = []
        for triple, label in data:
            triples.append(triple)
            labels.append(label)
        triple = torch.stack(triples, dim=0)
        trp_label = torch.stack(labels, dim=0)
        return triple, trp_label

    #for edges that exist in the graph, the entry is 1.0, otherwise the entry is 0.0
    def get_label(self, label, sample_type):
        # Currently no need for checking sample type too!
        y = np.zeros([self.num_ent], dtype=np.float32)
        for e2 in label: 
            y[e2] = 1.0
        return torch.FloatTensor(y)

   
class TestDataset(Dataset):
    """
    Evaluation Dataset class.
    Parameters
    ----------
    triples: The triples used for evaluating the model
    num_ent: Number of entities in the knowledge graph
    Returns
    -------
    An evaluation Dataset class instance used by DataLoader for model evaluation
    """
    def __init__(self, triples, num_ent):
        self.triples = triples
        self.num_ent = num_ent

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = torch.LongTensor(ele['triple']), np.int32(ele['label'])
        label = self.get_label(label)

        return triple, label

    @staticmethod
    def collate_fn(data):
        triples = []
        labels = []
        for triple, label in data:
            triples.append(triple)
            labels.append(label)
        triple = torch.stack(triples, dim=0)
        label = torch.stack(labels, dim=0)
        return triple, label

    #for edges that exist in the graph, the entry is 1.0, otherwise the entry is 0.0
    def get_label(self, label):
        y = np.zeros([self.num_ent], dtype=np.float32)
        for e2 in label: 
            y[e2] = 1.0
        return torch.FloatTensor(y)

class createTriplesDataset(Dataset):
    def __init__(self, g, p_edges_num, unknown_id, lbl_smooth, num_workers, batch_size, device):
        self.g = g
        self.p_edges_num = p_edges_num
        self.sample_size = 0.7 * p_edges_num
        self.pos_sampler = GlobalUniform_forTriples(self.g, self.sample_size, p_edges_num)
        self.lbl_smooth = lbl_smooth
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_rels = max(self.g.edata['etype']) + 1
        self.num_rels = unknown_id + 1

        eids, normal_eids = self.pos_sampler.sample(return_removed=True)
        eids = eids.to(device)
        normal_eids = normal_eids.to(device)
        # pos_rels = g.edata['etype'][eids]
        g.edata['etype'][eids] = unknown_id

        src, dst = g.find_edges(eids)
        # src, dst = src.numpy(), dst.numpy()
        # src_trip = src[th.where(g.edata['etype'][eids]==unknown_id)]
        # dst_trip = dst[th.where(g.edata['etype'][eids]==unknown_id)]
        # in_edges_mask = [True] * (self.g.num_edges()//2) + [False] * (self.g.num_edges()//2)
        # out_edges_mask = [False] * (self.g.num_edges()//2) + [True] * (self.g.num_edges()//2)
        # self.g.edata['in_edges_mask'] = torch.Tensor(in_edges_mask)
        # self.g.edata['out_edges_mask'] = torch.Tensor(out_edges_mask)
        
        #Opti
        self.triples = []
        for idx, e in enumerate(eids):
            self.triples.append({'triple':(src[idx], unknown_id, dst[idx]), 'label': g.edata['etype'][e].cpu().numpy(), 'ground_truth': th.tensor(1)})

        # put in negatives
        neg_eids = np.arange(p_edges_num, self.g.num_edges())
        src, dst = g.find_edges(neg_eids)
        for idx, e in enumerate(zip(src, dst)):
            self.triples.append({'triple':(e[0], unknown_id, e[1]), 'label': unknown_id, 'ground_truth': th.tensor(2)})
        
        src, dst = g.find_edges(normal_eids)
        for idx, e in enumerate(zip(src, dst, normal_eids)):
            self.triples.append({'triple':(e[0], g.edata['etype'][e[2]], e[1]), 'label': g.edata['etype'][e[2]].cpu().numpy(), 'ground_truth': th.tensor(0)})



    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label, gt = torch.LongTensor(ele['triple']), np.int32(ele['label']), ele['ground_truth']
        trp_label = self.get_label(label)
        #label smoothing
        if self.lbl_smooth != 0.0:
            trp_label = (1.0 - self.lbl_smooth) * trp_label + (1.0 / self.g.num_nodes())

        return triple, trp_label, gt
    
    @staticmethod
    def collate_fn(data):
        triples = []
        labels = []
        gts = []
        negatives = []
        positives = []
        i = 0
        for triple, label, gt in data:
            triples.append(triple)
            labels.append(label)
            gts.append(gt)
            if gt == 2:
                negatives.append(i)
            elif gt == 1:
                positives.append(i)
            i+=1

        triple = torch.stack(triples, dim=0)
        trp_label = torch.stack(labels, dim=0)
        gts = torch.stack(gts, dim=0)
        negatives = th.tensor(negatives)
        positives = th.tensor(positives) 
        return triple, trp_label, gts, negatives, positives
        
    #for edges that exist in the graph, the entry is 1.0, otherwise the entry is 0.0
    def get_label(self, label):
        y = np.zeros([self.num_rels], dtype=np.float32)
        y[label] = 1.0
        return torch.FloatTensor(y)



    

    # dataset, lbl_smooth, num_workers, batch_size, iep, edge_sampler):
    


class Data(object):

    def __init__(self, dataset, lbl_smooth, num_workers, batch_size, iep, edge_sampler):
        """
        Reading in raw triples and converts it into a standard format. 
        Parameters
        ----------
        dataset:           The name of the dataset
        lbl_smooth:        Label smoothing
        num_workers:       Number of workers of dataloaders
        batch_size:        Batch size of dataloaders
        Returns
        -------
        self.ent2id:            Entity to unique identifier mapping
        self.rel2id:            Relation to unique identifier mapping
        self.id2ent:            Inverse mapping of self.ent2id
        self.id2rel:            Inverse mapping of self.rel2id
        self.num_ent:           Number of entities in the knowledge graph
        self.num_rel:           Number of relations in the knowledge graph
        self.g:                 The dgl graph constucted from the edges in the traing set and all the entities in the knowledge graph
        self.data['train']:     Stores the triples corresponding to training dataset
        self.data['valid']:     Stores the triples corresponding to validation dataset
        self.data['test']:      Stores the triples corresponding to test dataset
        self.data_iter:		The dataloader for different data splits
        """
        self.dataset = dataset
        self.lbl_smooth = lbl_smooth
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.iep = iep
        self.edge_sampler = edge_sampler

        #read in raw data and get mappings
        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in ['train', 'test', 'valid']:
            for line in open('./{}/{}.txt'.format(self.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.num_ent = len(self.ent2id)
        self.num_rel = len(self.rel2id) // 2

        #read in ids of subjects, relations, and objects for train/test/valid 
        self.data = ddict(list) #stores the triples
        sr2o = ddict(set) #The key of sr20 is (subject, relation), and the items are all the successors following (subject, relation)
        src=[]
        dst=[]
        rels = []
        inver_src = []
        inver_dst = []
        inver_rels = []

        for split in ['train', 'test', 'valid']:
            for line in open('./{}/{}.txt'.format(self.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                sub_id, rel_id, obj_id = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub_id, rel_id, obj_id))

                if split == 'train': 
                    sr2o[(sub_id, rel_id)].add(obj_id)
                    sr2o[(obj_id, rel_id+self.num_rel)].add(sub_id) #append the reversed edges
                    src.append(sub_id)
                    dst.append(obj_id)
                    rels.append(rel_id)
                    inver_src.append(obj_id)
                    inver_dst.append(sub_id)
                    inver_rels.append(rel_id+self.num_rel)

        #construct dgl graph
        src = src + inver_src
        dst = dst + inver_dst
        rels = rels + inver_rels
        self.g = dgl.graph((src, dst), num_nodes=self.num_ent)
        self.g.edata['etype'] = torch.Tensor(rels).long()
        self.inc = IncGraphMaker(self.iep, self.g, self.edge_sampler, lbl_smooth, num_workers, self.edge_sampler)
        self.inc_g_wo_rel, self.inc_rel = self.inc.get_inc_graph()        

        self.inc_g_w_rel = copy.deepcopy(self.inc_g_wo_rel)
        self.inc_g_w_rel.edata['etype'] = torch.Tensor(self.inc_rel).long()            

        #identify in and out edges
        in_edges_mask = [True] * (self.g.num_edges()//2) + [False] * (self.g.num_edges()//2)
        out_edges_mask = [False] * (self.g.num_edges()//2) + [True] * (self.g.num_edges()//2)
        self.g.edata['in_edges_mask'] = torch.Tensor(in_edges_mask)
        self.g.edata['out_edges_mask'] = torch.Tensor(out_edges_mask)

        #Prepare train/valid/test data
        self.data = dict(self.data)
        self.sr2o = {k: list(v) for k, v in sr2o.items()} #store only the train data

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel+self.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()} #store all the data
        self.triples  = ddict(list)

        # take the triples and alter them accordingly
        # likewise we can run rwt also in this way itself

        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)]})

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        self.triples = dict(self.triples)

        def get_train_data_loader(split, batch_size, shuffle=True):
            return  DataLoader(
                    TrainDataset(self.triples[split], self.num_ent, self.lbl_smooth),
                    batch_size      = batch_size,
                    shuffle         = shuffle,
                    num_workers     = max(0, self.num_workers),
                    collate_fn      = TrainDataset.collate_fn
                )

        def get_test_data_loader(split, batch_size, shuffle=True):
            return  DataLoader(
                    TestDataset(self.triples[split], self.num_ent),
                    batch_size      = batch_size,
                    shuffle         = shuffle,
                    num_workers     = max(0, self.num_workers),
                    collate_fn      = TestDataset.collate_fn
                )

        #train/valid/test dataloaders
        self.data_iter = {
            'train':        get_train_data_loader('train', self.batch_size),
            'valid_head':   get_test_data_loader('valid_head', self.batch_size),
            'valid_tail':   get_test_data_loader('valid_tail', self.batch_size),
            'test_head':    get_test_data_loader('test_head', self.batch_size),
            'test_tail':    get_test_data_loader('test_tail', self.batch_size),
        }
        
