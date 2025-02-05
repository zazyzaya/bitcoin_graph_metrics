from argparse import ArgumentParser
import time
from types import SimpleNamespace

import torch
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from loader import get_graph
from models.cls_embedder import CLSEmbedder

torch.set_num_threads(8)
HYPERPARAMS = SimpleNamespace(
    epochs =    100,
    lr =        0.001,
    hidden =    64,
    latent =    32,
    patience =  10
)

def eval(g, model: CLSEmbedder):
    preds = torch.sigmoid(model.predict(g))
    y_hat = preds[g.test_mask].round().squeeze()
    y = g.y[g.test_mask]

    return accuracy_score(y, y_hat), \
        f1_score(y, y_hat), \
        precision_score(y, y_hat), \
        recall_score(y, y_hat)

def train(g, model: CLSEmbedder):
    opt = Adam(model.parameters(), lr=HYPERPARAMS.lr)

    for e in range(HYPERPARAMS.epochs):
        model.train()

        st = time.time()
        opt.zero_grad()
        loss = model.forward(g, g.train_mask, g.y.unsqueeze(-1))
        loss.backward()
        opt.step()
        en = time.time()

        with torch.no_grad():
            model.eval()
            acc, f1, pr, re = eval(g, model)

        print(f'[{e}] Loss: {loss.item():0.4f} Acc: {acc:0.4f}, F1: {f1:0.4f}, Pr: {pr:0.4f}, Re: {re:0.4f} ({en-st:0.2f}s)')
        model.save(f'checkpoints/{model.gnn_type}-CLS.pt')

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--gnn', default='gcn')
    args = ap.parse_args()

    if args.gnn == 'gat':
        torch.set_num_threads(16)

    g = get_graph()

    num_neg = (g.y[g.train_mask] == 0).sum()
    num_pos = g.y[g.train_mask].sum()
    pos_weight = num_neg / num_pos

    model = CLSEmbedder(
        g.x.size(1),
        HYPERPARAMS.hidden,
        HYPERPARAMS.latent,
        model_type=args.gnn,
        pos_weight=pos_weight
    )
    train(g, model)