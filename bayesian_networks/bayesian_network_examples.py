
import networkx as nx
import numpy as np

import bayesian_network
from cpds import StaticCPD, CategoricalCPD, CategoricalDistribution

def simple_example():
    # create variables
    A = 'A'
    B = 'B'
    C = 'C'

    # build the graph
    G = nx.DiGraph()
    G.add_edges_from([(A,B), (B,C)]) # chain

    # build the cpds
    A_dist = CategoricalDistribution([.5,.5])
    A_cpd = StaticCPD(A, A_dist)
    B_dists = {0: CategoricalDistribution([.5,.5]), 1: CategoricalDistribution([.1,.9])}
    B_cpd = CategoricalCPD(B, [A], 2, B_dists)
    C_dists = {0: CategoricalDistribution([.1,.9]), 1: CategoricalDistribution([.8,.2])}
    C_cpd = CategoricalCPD(C, [B], 2, C_dists)

    # build the bayesian network
    bn = bayesian_network.BayesianNetwork(G, [A_cpd, B_cpd, C_cpd])

    # compute some stuff
    prob = bn.pdf({'A':1, 'B':1, 'C':1})
    print(prob)
    prob = bn.pdf({'A':1, 'B':1, 'C':0})
    print(prob)

if __name__ == '__main__':
    simple_example()