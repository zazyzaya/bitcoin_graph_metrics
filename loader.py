import torch

from globals import graph_file

def get_graph():
    graph = torch.load(graph_file)
    graph.x = torch.nn.functional.normalize(graph.x[:, 1:], p=2, dim=1)
    graph.x = torch.nan_to_num(graph.x)
    return graph