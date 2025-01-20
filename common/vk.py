import numpy as np
from common import graph_info as gr
import re



def _edge_info(path_to_graph, order, group, num):
    edge_info = gr.get_edge_info(path_to_graph, order, group, num)
    edge_info_line = edge_info.split('\n')

    edge_pattern = r"Edge \((.*?)\)"
    edge_compiled_pattern = re.compile(edge_pattern)

    label_pattern = r"label=\[\s*(.*?)\s*\]"
    label_compiled_pattern = re.compile(label_pattern)

    eps_pattern = r"eps=\[\s*(.*?)\s*\]"
    eps_compiled_pattern = re.compile(eps_pattern)
    edges = []
    labels = []
    epsilons = []

    for line in edge_info_line:
        # Check if the line contains "stat_type 1" and does not contain "stat_type 0"
        if "stat_type 1" in line:
            edge_match = edge_compiled_pattern.search(line)
            label_match = label_compiled_pattern.search(line)
            eps_match = eps_compiled_pattern.search(line)
            if edge_match and label_match and eps_match:
                edge_tuple = list(map(int, edge_match.group(1).split(',')))
                labels_list = list(map(int, label_match.group(1).split()))
                epsilons_list = list(map(int, eps_match.group(1).split()))
                edges.append(edge_tuple)
                labels.append(labels_list)
                epsilons.append(epsilons_list)

    return edges, labels, epsilons

def _vk_lines(path_to_graph, order, group, num):
    data = np.loadtxt(path_to_graph + f'/o{order}_g{group}_n{num}.graph', dtype=int)
    data = data.tolist()

    source_d = dict()
    target_d = dict()
    edges_unique = set()

    for i in data:
        edges_unique.add(i[0])
        edges_unique.add(i[1])
        if i[1] not in target_d:
            target_d[i[1]] = [i[0:1] + i[2:3]]
        else:
            target_d[i[1]].append(i[0:1] + i[2:3])
        if i[0] not in source_d.keys():
            source_d[i[0]] = [i[1:3]]
        else:
            source_d[i[0]].append(i[1:3])

    unique_sources = list(edges_unique.difference(set(target_d)))
    unique_targets = list(edges_unique.difference(set(source_d)))
    if len(unique_sources) != 1:
        raise ValueError("More than one source edges detected")
    if len(unique_targets) != 1:
        raise ValueError("More than one target edges detected")

    vk1 = [source_d[unique_sources[0]][0][0]]
    vk1.extend([i[0] for i in source_d[vk1[0]] if i[1] == 1])

    vk2 = [target_d[unique_targets[0]][0][0]]
    vk2.extend([i[0] for i in target_d[vk2[0]] if i[1] == 1])
    vk2 = vk2[::-1]

    return vk1, vk2

def get_index_vk(path_to_graph, order, group, num):
    edges, labels, epsilons = _edge_info(path_to_graph, order, group, num)
    vk1, vk2 = _vk_lines(path_to_graph, order, group, num)
    vk = [edges.index(vk1), edges.index(vk2)]
    return vk