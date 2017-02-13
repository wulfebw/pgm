
import collections
import numpy as np

from factors import TabularFactor, LinearGaussianFactor
from markov_network import MarkovNetwork, Variable, Graph

def rmb_example():
    # create variables
    num_units = 5
    visible = [Variable('V{}'.format(i),[0,1]) for i in range(num_units)]
    hidden = [Variable('H{}'.format(i),[0,1]) for i in range(num_units)]
    variables = visible + hidden

    # create graph
    edges = collections.defaultdict(list)
    for v in visible:
        for h in hidden:
            edges[v].append(h)
            edges[h].append(v)
    graph = Graph(edges)

    # create pairwise factors
    cliques = []
    for v in visible:
        for h in hidden:
            cliques.append((h,v))
    factors = {}
    for clique in cliques:
        factors[clique] = TabularFactor(clique, np.array([[1, 1],[1, 1]]))

    # rbm 
    rbm = MarkovNetwork(graph, factors)
    print(rbm.partition)

if __name__ == '__main__':
    rmb_example()
