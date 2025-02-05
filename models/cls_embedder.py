import torch
from torch import nn
from torch_geometric.nn.models import GCN, GraphSAGE, GAT

class CLSEmbedder(nn.Module):
    def __init__(self, in_dim, hidden, emb_dim, pos_weight=1, layers=2, model_type='gcn'):
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
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.args = (in_dim, hidden, emb_dim)
        self.kwargs = dict(layers=layers, model_type=model_type, pos_weight=pos_weight)
        self.gnn_type = model_type

    def embed(self, x,ei):
        return self.gnn(x, ei)

    def predict(self, g):
        z = self.embed(g.x, g.edge_index)
        pred = self.out(z)
        return pred

    def forward(self, g, mask, labels):
        pred = self.predict(g)
        loss = self.criterion(pred[mask], labels[mask])

        return loss

    def save(self, fname):
        sd = self.state_dict()
        torch.save((self.args, self.kwargs, sd), fname)