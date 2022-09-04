import argparse
import torch as th
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


import dgl.function as fn
from dgl.dataloading import GraphDataLoader
import copy
import dgl

from utils import in_out_norm, inc_in_out_norm

from models_hgt_relation_prediction import CompGCN_ConvE
from data_loader_subgraph import Data, SubgraphIterator, IncGraphMaker, GlobalUniform, makeGraphFromEids, createTriplesDataset
import numpy as np
from time import time
import os
from dgl import KHopGraph


import debugpy
debugpy.listen(5950)
print("Waiting for debugger")
debugpy.wait_for_client()
print("Attached! :)")

#predict the tail for (head, rel, -1) or head for (-1, rel, tail)
def predict(model, graph, device, data_iter, split='valid', mode='tail'):
    model.eval()
    with th.no_grad():
        results = {}
        train_iter = iter(data_iter['{}_{}'.format(split, mode)])
        
        for step, batch in enumerate(train_iter):
            triple, label = batch[0].to(device), batch[1].to(device)
            sub, rel, obj, label = triple[:, 0], triple[:, 1], triple[:, 2], label
            pred = model(graph, sub, rel, obj)
            b_range = th.arange(pred.size()[0], device = device)
            target_pred = pred[b_range, obj]
            pred = th.where(label.byte(), -th.ones_like(pred) * 10000000, pred)
            pred[b_range, obj] = target_pred

            #compute metrics
            ranks = 1 + th.argsort(th.argsort(pred, dim=1, descending=True), dim =1, descending=False)[b_range, obj]
            ranks = ranks.float()
            results['count'] = th.numel(ranks) + results.get('count', 0.0)
            results['mr'] = th.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = th.sum(1.0/ranks).item() + results.get('mrr', 0.0)
            for k in [1,3,10]:
                results['hits@{}'.format(k)] = th.numel(ranks[ranks <= (k)]) + results.get('hits@{}'.format(k), 0.0)
            
    return results

#evaluation function, evaluate the head and tail prediction and then combine the results
def evaluate(model, graph, device, data_iter, split='valid'):
    #predict for head and tail
    left_results = predict(model, graph, device, data_iter, split, mode='tail')
    right_results = predict(model, graph, device, data_iter, split, mode='head')
    results = {}
    count = float(left_results['count'])

    #combine the head and tail prediction results
    #Metrics: MRR, MR, and Hit@k
    results['left_mr'] = round(left_results['mr']/count, 5)
    results['left_mrr'] = round(left_results['mrr']/count, 5)
    results['right_mr'] = round(right_results['mr']/count, 5)
    results['right_mrr'] = round(right_results['mrr']/count, 5)
    results['mr'] = round((left_results['mr'] + right_results['mr']) /(2*count), 5)
    results['mrr'] = round((left_results['mrr'] + right_results['mrr']) /(2*count), 5)
    for k in [1,3,10]:
        results['left_hits@{}'.format(k)] = round(left_results['hits@{}'.format(k)]/count, 5)
        results['right_hits@{}'.format(k)] = round(right_results['hits@{}'.format(k)]/count, 5)
        results['hits@{}'.format(k)] = round((left_results['hits@{}'.format(k)] + right_results['hits@{}'.format(k)])/(2*count), 5)
    return results 
    

def main(args):

    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # check cuda
    if args.gpu >= 0 and th.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    # embed = th.from_numpy(np.load("transe_embed/FB15k_TransE_l1_entity.npy")).to(device)
    # rel = th.from_numpy(np.load("transe_embed/FB15k_TransE_l1_relation.npy")).to(device)
    
    #construct graph, split in/out edges and prepare train/validation/test data_loader
    # This is the starting graph

    # 100% main graph --> data.g
    data = Data(args.dataset, args.lbl_smooth, args.num_workers, args.batch_size, args.iep, args.edge_sampler)
    data_iter = data.data_iter #train/validation/test data_loader
    #graph = data.gem.to(device)

    # data.inc_g_w_rel = data.inc_g_w_rel.to(device)
    
    # Creating K Hop edges
    # transform = KHopGraph(3)
    # khop_graph = transform(data.inc_g_wo_rel)

    #common after graph addition
    # khop_iep = min((0.7 * khop_mf * data.inc_g_wo_rel.num_edges()) / khop_graph.num_edges(), 1)
    # khop_graph_masked, _ = IncGraphMaker(iep=khop_iep, g=khop_graph, pos_sampler='uniform', lbl_smooth=args.lbl_smooth, num_workers=args.num_workers).get_inc_graph(has_etype=False)
    # num_rel_khop = th.max(data.g.edata['etype']).item() + 1
    # khop_graph_masked.edata["etype"] = torch.Tensor([num_rel_khop]*khop_graph_masked.num_edges()).long()

    # merged_graph = dgl.merge([data.inc_g_w_rel, khop_graph_masked])
    # add edge data in merged graph

    # mg_inc_ids, mg_inc_masked_ids = GlobalUniform(data.inc_g_w_rel, sample_size = 0.3 * data.inc_g_w_rel.num_edges()).sample(return_removed=True)
    # mg_inc_g = makeGraphFromEids(data.inc_g_wmg_rel, mg_inc_ids)

    # mg_khop_ids, mg_khop_ids_masked = GlobalUniform(khop_graph_masked, sample_size = 0.3 * khop_graph_masked.num_edges()).sample(return_removed=True)
    # mg_khop_g = makeGraphFromEids(khop_graph_masked, mg_khop_ids)
    # merged_graph = dgl.merge([mg_inc_g, mg_khop_g])

    #merged_graph = dgl.merge([khop_graph_masked, data.inc_g_w_rel])
    #num_rel = th.max(graph.edata['etype']).item() + 1

    # merged_graph_masked = IncGraphMaker(iep=0.3, g=khop_graph, pos_sampler='uniform', lbl_smooth=args.lbl_smooth, num_workers=args.num_workers).get_inc_graph(has_etype=False)
    

    # now we have a full graph, but we also want a graph with edge mask
    # full graph will be g and graph with edge mask will be gem
    # Shardul change - push to dgl subgraph only after proper subiteration
    #graph = data.g.to(device)


    # num relations stays the same
    # Shardul change
    graph = data.g
    num_rel = th.max(graph.edata['etype']).item() + 1

    #Compute in/out edge norms and store in edata
    graph = in_out_norm(graph)

    # Step 2: Create model =================================================================== #

    
    compgcn_model=CompGCN_ConvE(num_bases=args.num_bases,
                                num_rel=num_rel,
                                num_ent=graph.num_nodes(),
                                in_dim=args.init_dim,
                                layer_size=args.layer_size,
                                comp_fn=args.opn,
                                batchnorm=True,
                                dropout=args.dropout,
                                layer_dropout=args.layer_dropout,
                                num_filt=args.num_filt,
                                hid_drop=args.hid_drop,
                                feat_drop=args.feat_drop,
                                ker_sz=args.ker_sz,
                                k_w=args.k_w,
                                k_h=args.k_h,
                                emb=None,
                                rel=None
                                )

    # Shardul change - should be on device
    compgcn_model = compgcn_model.to(device)

    # Step 3: Create training components ===================================================== #
    loss_fn = th.nn.CrossEntropyLoss()
    loss_fn2 = th.nn.BCELoss()
    if args.optim == 'AdamW':
        print("Using AdamW as optimizer")
        optimizer = optim.AdamW(compgcn_model.parameters())
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, total_steps=294*args.max_epochs, max_lr = 1e-3, pct_start=0.05)
    else:
        optimizer = optim.Adam(compgcn_model.parameters(), lr=args.lr, weight_decay=args.l2)
    

    # Step 4: training epoches =============================================================== #
    best_mrr = 0.0
    kill_cnt = 0
    if args.optim == 'AdamW':
        train_step = 0
        steps_per_epoch = 0
    
    os.makedirs(f"checkpoints/{args.run_name}", exist_ok=True)

    # init_g = data.inc_g_wo_rel
    init_g_with_rel = data.inc_g_w_rel
    gamma = 0.8
    k = 2

    for epoch in range(args.max_epochs):
        if (epoch % args.add_after_epoch == 0):
             # Creating K Hop edges
            init_g = copy.deepcopy(init_g_with_rel)
            init_g.edata['etype'] = th.tensor([0] * init_g_with_rel.num_edges()).long()
            transform = KHopGraph(3)
            khop_graph = transform(init_g)

            #common after graph addition
            khop_iep = min((0.7 * args.khop_mf * init_g.num_edges()) / khop_graph.num_edges(), 1)
            khop_graph_masked, _ = IncGraphMaker(iep=khop_iep, g=khop_graph, pos_sampler='uniform', lbl_smooth=args.lbl_smooth, num_workers=args.num_workers).get_inc_graph(has_etype=False)
            num_rel_khop = num_rel
            khop_graph_masked.edata["etype"] = th.Tensor([num_rel_khop] * khop_graph_masked.num_edges()).long()

            merged_graph = dgl.merge([init_g_with_rel, khop_graph_masked])
            merged_graph = merged_graph.to(device)
            merged_graph = inc_in_out_norm(merged_graph)

            dataset = createTriplesDataset(merged_graph, init_g_with_rel.num_edges(), num_rel_khop, args.lbl_smooth, args.num_workers, args.batch_size, device)

            train_iter =   DataLoader(dataset, batch_size=args.batch_size, shuffle=True,collate_fn=dataset.collate_fn)

        # now run loop for training    
        # Training and validation using a full graph
        compgcn_model.train()
        train_loss=[]
        t0 = time()


        # for step, batch in enumerate(train_iter):


        # # no worries here since batch size is 1
        # for step_main, batch_main in enumerate(train_dataloader):
        #     sub_g, uniq_v, num_nodes, subgraph_data_iter, global_ids = batch_main
        #     # pushing graph on to device
        #     global_ids = th.tensor(global_ids).to(device)
        #     sub_g = sub_g.to(device)
        #     sub_g = in_out_norm(sub_g)
        #     #uniq_v = uniq_v.to(device)

        #     # looping over train data_loader for this graph
        #     for step, batch in enumerate(subgraph_data_iter['train']):
        #         triple, label = batch[0].to(device), batch[1].to(device)
        #         sub, rel, obj, label = triple[:, 0], triple[:, 1], triple[:, 2], label
        #         logits = compgcn_model(sub_g, sub, rel, global_ids)
        #          # compute loss
        #         tr_loss = loss_fn(logits, label)
        #         train_loss.append(tr_loss.item())

        merged_graph = merged_graph.to(device)
        top_k = []
        for step, batch in enumerate(train_iter):
            triple, label, gt, neg_indices, pos_indices = batch[0].to(device), batch[1].to(device), batch[2], batch[3].to(device), batch[4].to(device) # bs x 3, bs x num_rels + 1, 
            sub, rel, obj, label = triple[:, 0], triple[:, 1], triple[:, 2], label


            # logits = compgcn_model(merged_graph, sub, rel) # shape is 1024, 14541
            # negatives = gt[gt=="negative"]
            # positives = gt[gt=="positive"]
            logits = compgcn_model(merged_graph, sub, rel, obj)


            #####################################

            # compute loss
            tr_pos_loss = loss_fn(logits[pos_indices], label[pos_indices][:,:-1])  # classification loss

            # tr_pos_exist_loss = th.nn.Functional.Sigmoid(node_features[sub[pos_indices]] @ node_features[obj[pos_indices]])
            # tr_neg_exist_loss = -1 * th.sum(node_features[sub[pos_indices]] * node_features[obj[pos_indices]], dim=1)
            # tr_neg_loss = loss_fn(logits[neg_indices], label[neg_indices])

            logits_relation = th.max(logits, dim=1).indices
            logits_confidence = th.max(logits, dim=1).values

            neg_labels = th.zeros(neg_indices.shape).to(device)
            loss2 = loss_fn2(logits_confidence[neg_indices], neg_labels)

            tr_loss = gamma * tr_pos_loss + (1-gamma) * loss2

            train_loss.append(tr_loss.item())

            top_k.extend(sorted(zip(logits_confidence[neg_indices], logits_relation[neg_indices], sub[neg_indices], obj[neg_indices]), reverse=True)[:k])





            ######################################

            # backward
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()
            if args.optim == 'AdamW':
                train_step += 1
                steps_per_epoch += 1                
                scheduler.step(train_step)
                # print("Train Step:", train_step)
        
        if (epoch % args.add_after_epoch == 0):
            augment_edges = sorted(top_k, reverse=True)[:k]
            _, relations, sources, destinations = zip(*augment_edges)
            relations = th.tensor(relations)
            sources = th.tensor(sources)
            destinations = th.tensor(destinations)
            print(relations)

            init_g_with_rel.add_edges(sources, destinations)
            init_g_with_rel.edata['etype'][th.arange(len(init_g_with_rel.edata['etype']) - k, len(init_g_with_rel.edata['etype']))]  = relations

        train_loss = np.sum(train_loss)

        print(train_loss)
        continue
        t1 = time()  
        val_results = evaluate(compgcn_model, graph, device, data_iter, split='valid')
        t2 = time()

        #validate
        if val_results['mrr']>best_mrr:
            best_mrr = val_results['mrr']
            best_epoch = epoch
            th.save(compgcn_model.state_dict(), f"checkpoints/{args.run_name}/comp_link_best_model"+'_'+args.dataset)
            kill_cnt = 0
            print("saving model...")
        else:
            kill_cnt += 1
            if kill_cnt > 100:
                print('early stop.')
                break
        print("In epoch {}, Train Loss: {:.4f}, Valid MRR: {:.5}\n, Train time: {}, Valid time: {}"\
                .format(epoch, train_loss, val_results['mrr'], t1-t0, t2-t1))

        if epoch % 50 == 0:
            th.save(compgcn_model.state_dict(), f"checkpoints/{args.run_name}/comp_link_{epoch}_{val_results['mrr']}"+'_'+args.dataset)
            
    #test use the best model
    compgcn_model.eval()
    compgcn_model.load_state_dict(th.load(f"checkpoints/{args.run_name}/comp_link_best_model"+'_'+args.dataset))
    test_results = evaluate(compgcn_model, graph, device, data_iter, split='test')
    print("Test MRR: {:.5}\n, MR: {:.10}\n, H@10: {:.5}\n, H@3: {:.5}\n, H@1: {:.5}\n"\
            .format(test_results['mrr'], test_results['mr'], test_results['hits@10'], test_results['hits@3'], test_results['hits@1']))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument('--data', dest='dataset', default='FB15k-237', help='Dataset to use, default: FB15k-237')
    parser.add_argument('--data', dest='dataset', default='wn18rr', help='Dataset to use, default: FB15k-237')

    parser.add_argument('--model', dest='model', default='compgcn', help='Model Name')
    parser.add_argument('--score_func', dest='score_func', default='conve', help='Score Function for Link prediction')
    parser.add_argument('--opn', dest='opn', default='ccorr', help='Composition Operation to be used in CompGCN')

    parser.add_argument('--batch', dest='batch_size', default=1024, type=int, help='Batch size')
    parser.add_argument('--gpu', type=int, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--epoch', dest='max_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='Starting Learning Rate')
    parser.add_argument('--lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of processes to construct batches')
    parser.add_argument('--seed', dest='seed', default=41504, type=int, help='Seed for randomization')

    parser.add_argument('--num_bases', dest='num_bases', default=-1, type=int, help='Number of basis relation vectors to use')
    parser.add_argument('--init_dim', dest='init_dim', default=100, type=int, help='Initial dimension size for entities and relations')
    parser.add_argument('--layer_size', nargs='?', default='[200]', help='List of output size for each compGCN layer')
    parser.add_argument('--gcn_drop', dest='dropout', default=0.1, type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('--layer_dropout', nargs='?', default='[0.3]', help='List of dropout value after each compGCN layer')

    # ConvE specific hyperparameters
    parser.add_argument('--hid_drop', dest='hid_drop', default=0.3, type=float, help='ConvE: Hidden dropout')
    parser.add_argument('--feat_drop', dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('--k_w', dest='k_w', default=10, type=int, help='ConvE: k_w')
    parser.add_argument('--k_h', dest='k_h', default=20, type=int, help='ConvE: k_h')
    parser.add_argument('--num_filt', dest='num_filt', default=200, type=int, help='ConvE: Number of filters in convolution')
    parser.add_argument('--ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')

    parser.add_argument('--run_name', dest='run_name', default="default")

    # Add parameters for training
    parser.add_argument('--optim', dest='optim', default='Adam',  help='Optimizer name')
    parser.add_argument('--sched', dest='sched', default='', help='Scheduler name')
    
    # Parameters for graph sampling
    parser.add_argument("--edge-sampler", dest='edge_sampler', type=str, default='uniform',
                        choices=['uniform', 'neighbor'],
                        help="Type of edge sampler: 'uniform' or 'neighbor'"
                             "The original implementation uses neighbor sampler.")

    parser.add_argument("--initial_edge_percentage", dest='iep', type=float, default='0.1')
    parser.add_argument("--k_hop_list", nargs='?', default='[1, 2, 3]', dest='khl')
    parser.add_argument("--add_after_epoch", type=int, default=2, dest='add_after_epoch')
    parser.add_argument("--khop_mf", type=float, default=10, dest='khop_mf')


    args = parser.parse_args()
    
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    print(args)
    

    args.layer_size = eval(args.layer_size)
    args.layer_dropout = eval(args.layer_dropout)

    main(args)
    