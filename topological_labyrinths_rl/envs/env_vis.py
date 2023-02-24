from matplotlib import pyplot as plt

from graph_mdp import GraphEnv2D
import networkx as nx


class Visualizer:
    def __init__(self, graph_mdp: GraphEnv2D, node_color='#1f78b4', node_size=300):
        self.graph_mdp = graph_mdp
        self.n_states = graph_mdp.n_states
        self.graph_adjacency = graph_mdp.graph_adjacency
        self.node_positions = graph_mdp.pos

        self.node_color = node_color
        self.node_size = node_size

    def nx_draw_graph(self):
        node_colors = [self.node_color] * self.n_states
        node_sizes = [self.node_size] * self.n_states

        node_colors[self.graph_mdp.goal_state] = '#1f7554'

        node_colors[self.graph_mdp.state] = '#1f3224'
        node_sizes[self.graph_mdp.state] = 500

        nx_graph = nx.from_numpy_matrix(self.graph_adjacency)

        nx.draw(nx_graph, pos=self.node_positions, node_color=node_colors, node_size=node_sizes)
