import networkx as nx
import matplotlib.pyplot as plt

from qaoa_engine import QAOA
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from networkx.drawing.layout import spring_layout


class MaxCut(QAOA):

    """QAOA implementation for Max Cut."""

    def __init__(self, V, E):

        """
        Build input graph and begin QAOA
        Args:
            V: Vertices of input graph as a list.
            E: Edges of input graph as a dictionary with weights or a list if unweighted.
        """

        # Set up vertices and edges
        self.vertices = V
        if isinstance(E, dict):
            self.edges = list(E.keys())
            self.weights = [E[k] for k in self.edges]
        else:
            self.edges = E
            self.weights = [1]*len(self.edges)

        # Build input graph
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.vertices)
        self.graph.add_edges_from(self.edges)

        # Begin QAOA
        super().__init__(len(V), p=6)

    def cost_function(self, z):

        """
        MaxCut cost function.
        Args:
            z: An integer or bitstring whose cost is to be determined.
        Return:
            Cost function as integer.
        """

        # Convert to bitstr
        if not isinstance(z, str):
            z = format(z, '0' + str(self.n) + 'b')
        z: str

        # Evaluate dummy C(z)
        cost = len(z)
        return cost

    def build_cost_ckt(self):

        """
        Dummy cost circuit. Override in child class.
        Return:
            QuantumCircuit. Parameterized cost circuit layer.
        """

        # Build dummy circuit
        circ = QuantumCircuit(self.n, name='$U(H_C,\\gamma)$')
        circ.rz(Parameter('param_c'), range(self.n))
        return circ

    def visualize_output(self):

        """
        Visualize Max Cut output post QAOA optimization.
        """

        # Sample output
        z = self.sample(vis=True)
        print('Sampled Output: ' + str(z))
        print('Optimized Cost: ' + str(self.cost_function(z)))

        # Draw variational circuit
        self.variational_ckt.draw(reverse_bits=True, initial_state=True)\
            .suptitle('Variational Circuit', fontsize=16)

        # Extract colormap
        color_map = []
        for i in range(len(self.vertices)):
            if z[i] == '0':
                color_map.append('blue')
            else:
                color_map.append('red')

        # Extract cuts
        cuts = []
        for e in self.edges:
            if z[e[0]] == z[e[1]]:
                cuts.append('solid')
            else:
                cuts.append('dashed')

        # Draw input graph
        fig = plt.figure(figsize=(10, 5))
        fig.suptitle('Max Cut')
        plt.subplot(121)
        ax = plt.gca()
        ax.set_title('Input Graph')
        pos = spring_layout(self.graph)
        nx.draw(self.graph, with_labels=True, node_color='lightgreen', edge_color='lightblue',
                style='solid', width=2, ax=ax, pos=pos, font_size=8, font_weight='bold')

        # Draw output graph
        plt.subplot(122)
        ax = plt.gca()
        ax.set_title('Output Graph')
        nx.draw(self.graph, with_labels=True, node_color=color_map, edge_color='green',
                style=cuts, width=2, ax=ax, pos=pos, font_size=8, font_weight='bold')
        plt.show()
