# GNNProject_SHAP

## Abstract
As we all know, Graph neural networks(GNNs) will be able to work the best when the entire graph structure is provided. However, it is not practical in a real-world setting to have the entire graph structure provided. One of the earlier solutions proposed to this problem is that we devise a task-specific latent structure and apply a GNN to the inferred graph. One of the major possibilities of the space of the graph is that it grows super-exponentially with the number of nodes and so the proposed structure for task-specific will be inefficient for learning the structure and GNN parameters. Another
problem that arises when dealing with large graph structures is they are often incomplete. To deal with this, methods have proposed using similarity graphs based on the initial features or learning the graph structure and the latent representations simultaneously, with the latter method often achieving better results. Through this work we aim to propose a solution for completing the graph structure, leveraging transformers attention technique to learn better representations of the relations in graph, and iteratively building them upon the original graph. In end, we show some preliminary analysis as part of this project which motivated us to pursue this line of direction.


The main code lies inside CompGCN+ATT folder.

The dataset can be downloaded using the .sh files available.

Requirements:
torch
dgl-cu102 (depends on cuda version)

To run the training for CompGCN + ATT on full graph run: (Can change the data)
```
python main_hgt_base.py --score_func conve --opn ccorr --gpu 0  --data FB15k-237 --num_bases 50 —optim AdamW
```


To run the training for incomplete graph run

```
python main_ablation.py --score_func conve --opn ccorr  --num_bases 5 --optim AdamW --initial_edge_percentage 0.8  --run_name ablation_80p_hgt_conve_corr_base --gpu  6 —data wn18rr

```

To run the training For Structural Graph Learning run

```
python main_khop.py --score_func conve --opn ccorr --gpu 0  --num_bases 5 --run_name temp --data wn18rr
```
