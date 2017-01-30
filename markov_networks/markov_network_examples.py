
import copy 
import numpy as np

from factors import TabularFactor, LinearGaussianFactor
from markov_network import MarkovNetwork, Variable, Graph


def binary_example():
    """
    three variable markov network with factors associated with the binary 
    cliques, of which there are three
    """
    # create the variables
    A = Variable('A', [0,1])
    B = Variable('B', [0,1])
    C = Variable('C', [0,1])

    # create graph
    edges = {}
    edges[A] = [B, C]
    edges[B] = [A, C]
    edges[C] = [A, B]
    graph = Graph(edges)

    # create pairwise factors
    cliques = [(A,B), (A,C), (B,C)]
    factors = {}
    for clique in cliques:
        factors[clique] = TabularFactor(clique, np.array([[10, 1],[1, 10]]))

    # create the markov network
    mn_pairwise = MarkovNetwork(graph, factors)

    # test
    scaled_factors = copy.deepcopy(factors)
    scaled_factors[(A,B)] = TabularFactor((A,B), 10 * np.array([[10, 1],[1, 10]]))

    # create the markov network
    mn_pairwise_scaled = MarkovNetwork(graph, factors)

    # create the factor over all the variables
    factors = {(A,B,C):
        TabularFactor((A,B,C), np.array([
            [
                [1000,10],
                [10,10]
            ], 

            [
                [10,10],
                [10,1000]
            ]
        ]))
    }

    # create the markov network
    mn_single = MarkovNetwork(graph, factors)

    # what's the probability of the assignment (A:0, B:0, C:0)?
    assignment = {A:0, B:0, C:0}
    prob = mn_pairwise.probability(assignment)
    print(prob)
    affinity = mn_pairwise._assignment_affinity(assignment)
    print(affinity)
    
    print()
    prob = mn_single.probability(assignment)
    print(prob)
    affinity = mn_single._assignment_affinity(assignment)
    print(affinity)

def haircolor_example():
    # create the variables
    num_colors = 3
    A = Variable('A', [0,1,2])
    B = Variable('B', [0,1,2])
    C = Variable('C', [0,1,2])
    D = Variable('D', [0,1,2])

    # create graph
    edges = {}
    edges[A] = [B, D]
    edges[B] = [A, C]
    edges[C] = [B, D]
    edges[D] = [A, C]
    graph = Graph(edges)

    # create factors
    cliques = [(A,B), (B,C), (C,D), (D,A), (A,), (B,), (C,), (D,)]
    factors = [1 - np.eye(num_colors)] * 4
    factors += [[1,1,1], [1,1,1], [1,1,1], [1,1,1]] 

    # D doesn't like the first color at all and likes the third a lot
    factors[-1][0] = 0
    factors[-1][2] = 10
    factors = {c:TabularFactor(c,f) for (c,f) in zip(cliques, factors)}
    
    # create the markov network
    mn = MarkovNetwork(graph, factors)

    print(mn.partition)
    # 0 b/c D doesn't like color 0
    assignment = {A:1, B:0, C:1, D:0}
    print(mn.p(assignment))
    # large b/c D does like color 2
    assignment = {A:1, B:0, C:1, D:2}
    print(mn.p(assignment))
    # 0 b/c D and C cannot have the same color
    assignment = {A:1, B:0, C:2, D:2}
    print(mn.p(assignment))

    # what assignment of the variables is most likely?
    ml_assignment = mn.most_likely_assignment()
    print(ml_assignment)

    # is A conditionally independent of B given C and D? no
    indp = mn.conditionally_independent(set([A]), set([B]), set([C,D]))
    print(indp)
    # how about A indp of C given B and D?
    indp = mn.conditionally_independent(set([A]), set([C]), set([B,D]))
    print(indp)

def linear_gaussian_example():
    # create the variables (the largest affinity will be when they sum to 0)
    A = Variable('A', [-1,0,1])
    B = Variable('B', [-1,0,1])

    # create graph
    edges = {}
    edges[A] = [B]
    edges[B] = [A]
    graph = Graph(edges)

    # create factors
    cliques = [(A,B)]
    factors = {
        (A,B): LinearGaussianFactor((A,B)) # default is unit gaussian
    }

    # create the markov network
    mn = MarkovNetwork(graph, factors)

    # what's the partition function
    print(mn.partition)

    # what assignment of the variables is most likely?
    ml_assignment = mn.most_likely_assignment()
    print(ml_assignment)
    # what's the probability of that assignment?
    print(mn.p(ml_assignment))

if __name__ == '__main__':
    binary_example()
    haircolor_example()
    linear_gaussian_example()
