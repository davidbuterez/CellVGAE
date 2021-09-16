import torch

def extract_attn_data(edge_index_attn, weights_attn, dim=None, k=120):
    edges = edge_index_attn.T
    if not dim:
        w = weights_attn.mean(dim=1)
    else:
        w = weights_attn[:, dim]
    w = w.squeeze()
    top_values, top_indices = torch.topk(w, k)
    top_edges = edges[top_indices]
    
    return top_edges, top_values