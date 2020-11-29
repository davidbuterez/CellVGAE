import torch
import copy
import cairo
import pandas as pd
import numpy as np
import seaborn as sns

from graph_tool.all import *

def build_attn_graph(edge_index_attn, weights_attn, dim=None, k=120):
    edges = edge_index_attn.T
    if not dim:
        w = weights_attn.mean(dim=1)
    else:
        w = weights_attn[:, dim]
    w = w.squeeze()
    top_values, top_indices = torch.topk(w, k)
    top_edges = edges[top_indices]
    edge_data = np.concatenate([edges[top_indices].numpy(), np.expand_dims(top_values.detach().numpy(), 1)], axis=1)
    
    g = Graph()
    edge_weight = g.new_edge_property('double')

    # adding list of all properties 
    eprops = [edge_weight]

    # add edges and properties to the graph
    g.add_edge_list(edge_data, eprops=eprops)
    
    degrees = g.get_total_degrees(list(g.vertices()))
    degrees = torch.from_numpy(degrees.astype(np.int32))
    zero_degrees_indices = (degrees == 0).nonzero()
    nodes_to_remove = np.array(range(len(degrees)))[zero_degrees_indices]
    g.remove_vertex(vertex=nodes_to_remove)
    
    return g, nodes_to_remove, edge_weight


def plot_graph(g, nodes_not_included, cells, name, eprops_scaled):    
    number_to_cellname = dict(zip(range(len(cells)), cells))
    
    plot_cell_names = []
    for i in range(len(cells)):
        if i not in nodes_not_included:
            plot_cell_names.append(number_to_cellname[i])
    
    types = list(set(cells))
    palette = sns.color_palette("Set2", len(types))
    colors = []
    for cell_name in plot_cell_names:
        idx = types.index(cell_name)
        colors.append((palette[idx][0], palette[idx][1], palette[idx][2], 1))
            
    v_prop_cell_name = g.new_vertex_property('string', vals=plot_cell_names)
    plot_color = g.new_vertex_property('vector<double>', vals=colors)
    
    # Attention e_ij is influence of node j to node i
    g.set_reversed(True)
    
    layout_weights = copy.deepcopy(eprops_scaled)
    layout_weights = layout_weights.set_2d_array(layout_weights.a * 10)
    
    pos = arf_layout(g, weight=layout_weights, max_iter=0, d=10, a=2)

    # Optional: add arguments: output='%s.pdf' % (name), fmt='pdf'
    graph_draw(g, pos=pos, vertex_size=20, vertex_fill_color=plot_color, vertex_text=v_prop_cell_name, 
               vertex_font_family='Source Pro', vertex_font_size=7, vertex_text_position=-2, vertex_font_weight=cairo.FONT_WEIGHT_BOLD,
               nodesfirst=True, edge_pen_width=eprops_scaled)


def build_and_plot_attn_graph(attn_edge_index, attn_coeff, cells, name, dim, edge_scale_factor=8, k=80):
    g_attn, nodes_not_included_attn, eprops_attn = build_attn_graph(attn_edge_index, attn_coeff, dim, k=k)
    eprops_attn.set_2d_array(eprops_attn.a * edge_scale_factor)
    if dim:
        plot_graph(g_attn, nodes_not_included_attn, cells, name + '_%d' % (dim), eprops_attn)
    else:
        plot_graph(g_attn, nodes_not_included_attn, cells, name + '_mean', eprops_attn)