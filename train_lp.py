from argparse import ArgumentParser
import time
from types import SimpleNamespace

import torch
from torch.optim import Adam

from loader import get_graph
from models.lp_embedder import LPEmbedder

torch.set_num_threads(8)
HYPERPARAMS = SimpleNamespace(
    epochs =    100,
    lr =        0.001,
    hidden =    64,
    latent =    32,
    patience =  10
)

def train(g, model: LPEmbedder):
    opt = Adam(model.parameters(), lr=HYPERPARAMS.lr)

    n_edges = g.edge_index.size(1)
    for e in range(HYPERPARAMS.epochs):
        # Randomly select 10% of edges to train on
        mask = torch.rand(n_edges) < 0.1

        ei = g.edge_index[:, ~mask]
        pos = g.edge_index[:, mask]
        neg = torch.randint(0, g.x.size(0), pos.size())

        st = time.time()
        opt.zero_grad()
        loss = model(g.x, ei, pos, neg)
        loss.backward()
        opt.step()
        en = time.time()

        print(f'[{e}] Loss: {loss.item():0.4f} ({en-st:0.2f}s)')
        model.save(f'checkpoints/{model.gnn_type}-LP.pt')

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--gnn', default='gcn')
    args = ap.parse_args()

    if args.gnn == 'gat':
        torch.set_num_threads(16)

    g = get_graph()
    model = LPEmbedder(
        g.x.size(1),
        HYPERPARAMS.hidden,
        HYPERPARAMS.latent,
        model_type=args.gnn
    )
    train(g, model)