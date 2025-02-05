import torch
from torch import nn
from torch_geometric.nn.models import GCN, GraphSAGE, GAT

class LPEmbedder(nn.Module):
    def __init__(self, in_dim, hidden, emb_dim, layers=2, model_type='gcn'):
        super().__init__()

        model_type = model_type.lower()
        assert model_type in ['gcn', 'sage', 'gat'], \
            'Model must be one of [GCN, SAGE, GAT]'

        gnn_constructor = {
            'gcn': GCN, 'sage': GraphSAGE, 'gat': GAT
        }[model_type]
        self.gnn = gnn_constructor(
            in_dim, hidden,
            layers, out_channels=emb_dim
        )

        self.out = nn.Linear(emb_dim, 1)
        self.criterion = nn.BCEWithLogitsLoss()

        self.args = (in_dim, hidden, emb_dim)
        self.kwargs = dict(layers=layers, model_type=model_type)
        self.gnn_type = model_type

    def embed(self, x,ei):
        return self.gnn(x, ei)

    def forward(self, x, ei, pos, neg):
        z = self.embed(x,ei)

        pos_preds = self.out(z[pos[0]] * z[pos[1]])
        neg_preds = self.out(z[neg[0]] * z[neg[1]])
        preds = torch.cat([pos_preds, neg_preds])

        labels = torch.zeros(pos.size(1)+neg.size(1), 1)
        labels[:pos.size(0)] = 1

        loss = self.criterion(preds, labels)
        return loss

    def save(self, fname):
        sd = self.state_dict()
        torch.save((self.args, self.kwargs, sd), fname)