import networkx as nx
import matplotlib.pyplot as plt

from itertools import combinations
from qaoa_engine import QAOA
from networkx.drawing.layout import spring_layout
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import depolarizing_error


class MaxClique(QAOA):

    """QAOA implementation for Max Clique."""

    def __init__(self, V, E, noise_model=None):

        """
        Build input graph and begin QAOA
        Args:
            V: Vertices of input graph as a list.
            E: Edges of input graph as a list.
            noise_model: Qiskit NoiseModel instance. Optional argument for noisy simulations.
        """

        # Set up vertices and edges
        self.vertices = V
        self.edges = E

        # Build input graph
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.vertices)
        self.graph.add_edges_from(self.edges)
        self.anti_edges = self.acquire_anti_graph()

        # Begin QAOA
        super().__init__(len(V), p=6, noise_model=noise_model)

    def cost_function(self, z):

        """
        Max Clique cost function.
        Args:
            z: An integer or bitstring whose cost is to be determined.
        Return:
            Cost function as integer.
        """

        # Convert to bitstr
        if not isinstance(z, str):
            z = format(z, '0' + str(self.n) + 'b')
        z: str

        # Evaluate C(z)
        cost = 0
        for i in range(self.n):
            cost = cost - 1 if z[i] == '1' else cost + 1
        for edge in self.anti_edges:
            cost = cost + 3 if z[edge[0]] == '1' and z[edge[1]] == '1' else cost - 1
        return cost

    def build_cost_ckt(self):

        """
        Max Clique cost circuit.
        Return:
            QuantumCircuit. Parameterized cost circuit layer.
        """

        # Build cost circuit
        param = Parameter('param_c')
        circ = QuantumCircuit(self.n, name='$U(H_C,\\gamma)$')
        circ.rz(2*param, range(self.n))
        for edge in self.anti_edges:
            circ.cp(-4*param, edge[0], edge[1])
        return circ

    def acquire_anti_graph(self):

        """
        Build the anti graph of the input graph.
        Return:
            List of tuples representing anti edges.
        """

        # Acquire anti edges
        anti_edges = []
        for edge in combinations(self.vertices, 2):
            if edge not in self.edges and edge[::-1] not in self.edges:
                anti_edges.append(edge)
        return anti_edges

    def visualize_output(self):

        """
        Visualize Max Clique output post QAOA optimization.
        """

        # Sample output
        z, avg_cost = self.sample(vis=True)
        print('Sampled Output: ' + str(z))
        print('Minimum Cost: ' + str(self.cost_function(z)))
        print('Expectation Value: ' + str(avg_cost))

        # Extract colormap
        color_map = []
        for i in range(len(self.graph.nodes)):
            if z[i] == '0':
                color_map.append('red')
            else:
                color_map.append('blue')

        # Extract cuts
        cuts = []
        for e in self.graph.edges:
            if z[e[0]] == '1' and z[e[1]] == '1':
                cuts.append('solid')
            else:
                cuts.append('dashed')

        # Draw input graph
        fig = plt.figure(figsize=(10, 5))
        fig.suptitle('Max Clique')
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


if __name__ == '__main__':

    # Test code
    p = 0.1
    V = list(range(7))
    E = list(combinations(range(1, 6), 2)) + [(0, 1), (0, 2), (6, 4), (6, 5)]
    noise_model = NoiseModel()
    error = depolarizing_error(p, num_qubits=1)
    noise_model.add_all_qubit_quantum_error(error, 'noise')
    noise_model.add_basis_gates(['unitary'])
    obj = MaxClique(V, E, noise_model=noise_model)
    obj.visualize_output()
