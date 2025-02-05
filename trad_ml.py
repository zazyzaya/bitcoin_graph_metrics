from rich import print
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import time
import torch
import pandas as pd
from xgboost import XGBClassifier

from loader import get_graph
from models.cls_embedder import CLSEmbedder
from models.lp_embedder import LPEmbedder

MODEL_DIR = 'checkpoints/100-epochs'
RESULT_DIR = 'results'

def eval(g, gnn_type, objective):
    objective = objective.upper()
    if objective == 'LP':
        Embedder = LPEmbedder
    elif objective == 'CLS':
        Embedder = CLSEmbedder
    else:
        raise ValueError("objective must be either 'CLS' or 'LP'")

    args,kwargs,sd = torch.load(f'{MODEL_DIR}/{gnn_type}-{objective}.pt')
    model = Embedder(*args, **kwargs)
    model.load_state_dict(sd)
    model.eval()

    with torch.no_grad():
        zs = model.embed(g.x, g.edge_index)

    x = torch.cat([g.x, zs], dim=1)
    y = g.y
    tr_mask = g.train_mask
    te_mask = g.test_mask

    metrics = []
    for Model in [XGBClassifier, RandomForestClassifier, DecisionTreeClassifier]:
        model = Model(n_jobs=32) # Todo: hyperparams

        print("Fitting", end='', flush=True)
        st = time.time()
        model.fit(x[tr_mask], y[tr_mask])
        print(f" ({time.time()-st:0.2f})")

        print("Predicting", end='', flush=True)
        st = time.time()
        y_hat = model.predict(x[te_mask])
        print(f" ({time.time()-st:0.2f})")

        print("Getting metrics", end='', flush=True)
        name = model.__class__.__name__
        acc = accuracy_score(y[te_mask], y_hat)
        bac = balanced_accuracy_score(y[te_mask], y_hat)
        pr = precision_score(y[te_mask], y_hat)
        re = recall_score(y[te_mask], y_hat)
        f1 = f1_score(y[te_mask], y_hat)
        cm = confusion_matrix(y[te_mask], y_hat)
        tps = cm[1,1]; fps = cm[0,1]
        tns = cm[0,0]; fns = cm[1,0]
        print(f" ({time.time()-st:0.2f})")

        metrics.append({
            'classifier': name,
            'gnn': gnn_type,
            'acc': acc, 'b-acc': bac,
            'pr': pr, 're': re, 'f1': f1,
            'tps': tps, 'fps': fps,
            'tns': tns, 'fns': fns
        })
        print(metrics[-1])

    return metrics

if __name__ == '__main__':
    g = get_graph()

    for objective in ['LP', 'CLS']:
        metrics = []
        for gnn in ['gcn', 'sage', 'gat']:
            print(f"{objective}, {gnn}")
            m = eval(g, gnn, objective)
            metrics.append(m)

        df = pd.DataFrame(metrics)
        df.to_csv(f'{RESULT_DIR}/{objective}.csv')
