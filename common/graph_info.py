import sys
import pathlib
from collections import defaultdict
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()
sys.path.append(str(pathlib.Path(__file__).parent.joinpath(os.getenv('CPLUS_INCLUDE_PATH')).resolve()))
import pytami

def nested_defaultdict():
    return defaultdict(nested_defaultdict)

def get_R0_prefactor_dict_ord(path_to_graph, orders: list):
    graph_type: pytami.TamiBase.graph_type = pytami.TamiBase.Sigma
    seed: int = 0

    graph: pytami.TamiGraph = pytami.TamiGraph(graph_type, seed)

    folder: str = path_to_graph # specify folder with graphs, path should be relative
    ggm: pytami.TamiGraph.gg_matrix_t = pytami.TamiGraph.gg_matrix_t()

    graph.read_ggmp(folder, ggm, orders[0], orders[-1])
    graph.ggm_label(ggm, 0)

    prefactor_dict = nested_defaultdict()
    R0_dict = nested_defaultdict()
    for ord in orders:
        for group in range(len(ggm[ord])):
            gg: pytami.TamiGraph.graph_group = ggm[ord][group].graph_vec
            for num in range(len(gg)):
                gg: pytami.TamiGraph.graph_group = ggm[ord][group].graph_vec
                diagram: pytami.TamiGraph.trojan_graph = pytami.TamiGraph.trojan_graph(gg, num)
                prefactor: float = graph.trojan_get_prefactor(diagram, ord)
                R0: pytami.TamiBase.g_prod_t = pytami.TamiBase.g_prod_t()
                graph.trojan_graph_to_R0(diagram, R0)
                R0_dict[ord][group][num] = R0
                prefactor_dict[ord][group][num] = prefactor

    return R0_dict, prefactor_dict

def get_R0_prefactor_specific(path_to_graph, ord, group, num):
    graph_type: pytami.TamiBase.graph_type = pytami.TamiBase.Sigma
    seed: int = 0

    graph: pytami.TamiGraph = pytami.TamiGraph(graph_type, seed)

    folder: str = path_to_graph # specify folder with graphs, path should be relative
    ggm: pytami.TamiGraph.gg_matrix_t = pytami.TamiGraph.gg_matrix_t()
    graph.read_ggmp(folder, ggm, ord, ord)
    graph.ggm_label(ggm, 0)

    gg: pytami.TamiGraph.graph_group = ggm[ord][group].graph_vec
    diagram: pytami.TamiGraph.trojan_graph = pytami.TamiGraph.trojan_graph(gg, num)  # this is ugly because of the boost graph library objects not being visible to python
    prefactor: float = graph.trojan_get_prefactor(diagram, ord)
    R0: pytami.TamiBase.g_prod_t = pytami.TamiBase.g_prod_t()
    graph.trojan_graph_to_R0(diagram, R0)
    return R0, prefactor

def print_eps_alpha_dict(R0_dict):
    for ord in R0_dict.keys():
        for group in R0_dict[ord].keys():
            for num, value in R0_dict[ord][group].items():
                print("Graph", f"o{ord}_g{group}_n{num}")
                for indx, v in enumerate(value):
                    print("Epsilon", indx, v.eps_)
                for indx, v in enumerate(value):
                    print("Alpha", indx, v.alpha_)

def print_eps_alpha(R0):
    for indx, v in enumerate(R0):
        print("Epsilon", indx, v.eps_)
    for indx, v in enumerate(R0):
        print("Alpha", indx, v.alpha_)

