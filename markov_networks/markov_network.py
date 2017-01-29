
import itertools
import numpy as np
import os
import sys

path = os.path.join(os.path.dirname(__file__), os.pardir, 'graphs')
sys.path.append(os.path.abspath(path))

import search_problem
import bfs

class Variable(object):
    """
    Description:
        - Variable object represents a variable in the graph.
    """
    def __init__(self, name, domain):
        self.name = name
        self.domain = domain
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        return self.name == other.name
    def __ne__(self, other):
        return not(self == other)
    def __repr__(self):
        return self.name

class Graph(object):
    """
    Description:
        - Graph represents the graph in a markov network, it acts as a structured
            collection of variables.
    """
    def __init__(self, edges):
        self.edges = edges
        self.variables = edges.keys()
        self.domains = [v.domain for v in self.variables]

    def assignments(self):
        """
        Description:
            - Generator of dictionaries, each of which is a unique assignment
                to the variables in the graph.
        """
        for values in itertools.product(*self.domains):
            yield {var:value for (var,value) in zip(self.variables, values)}

    def path(self, src, dest, absent=set()):
        """
        Description:
            - Determine whether there exists a path between the src and dest
                optionally without crossing any of the nodes in the absent set.

        Args:
            - src: set of nodes from which to begin the path
            - dest: set of nodes at which to complete the path
            - absent: set of nodes that cannot be used in the path
        """
        algorithm = bfs.BreadthFirstSearch()
        for s in src:
            problem = search_problem.UndirectedGraphSearchProblem(
                self.edges, s, dest, absent)
            path = algorithm.solve(problem)
            if path != []:
                return path
        return []

class Factor(object):
    """
    Description:
        - Factor represents a potential function over a set of variables.
    """

    def __init__(self, clique, table):
        self.clique = clique
        self.table = np.asarray(table)

    def affinity(self, assignment):
        """
        Description:
            - Return the affinity associated with an assignment, or 1 if the 
                assignment contains no variables associated with this factor.

        Args:
            - assignment: dict mapping variables to values
        """
        key = tuple(assignment[var] for var in self.clique)
        if not key:
            return 1
        else:
            return self.table[key]

class MarkovNetwork(object):

    def __init__(self, graph, factors):
        """
        Args:
            - graph: graph object
            - factors: dict mapping cliques of variables to Factor objects
        """
        self.graph = graph
        self.factors = factors
        self.partition = self._compute_partition_function()

    def probability(self, assignment):
        """
        Description:
            - Return the normalized probability associated with an assignment.

        Args:
            - assignment: dict mapping variables to values
        """
        return self._assignment_affinity(assignment) / self.partition

    def p(self, assignment):
        """
        Description:
            - Alias for markov_network.probability
        """
        return self.probability(assignment)

    def _assignment_affinity(self, assignment):
        """
        Description:
            - Return the total affinity associated with an assignment.

        Args:
            - assignment: dict mapping variables to values
        """
        assignment_total = 1
        for (clique, factor) in self.factors.items():
            assignment_total *= factor.affinity(assignment)
        return assignment_total

    def _compute_partition_function(self):
        """
        Description:
            - Compute the partition value by iterating over all possible
                assignments of variables in the graph. This is naive in that 
                it does not take advantage of the graph stucture at all, though 
                in general computing this value is expensive.
        """
        total = 0
        for assignment in self.graph.assignments():
            total += self._assignment_affinity(assignment)
        return total

    def most_likely_assignment(self):
        """
        Description:
            - Naive method for finding the most likely assignment of the 
                variables (by explictly computing them all and taking the 
                largest).
        """
        best_assignment, best_affinity = {}, 0
        for assignment in self.graph.assignments():
            affinity = self._assignment_affinity(assignment)
            if affinity > best_affinity:
                best_affinity, best_assignment = affinity, assignment
        return best_assignment

    def conditionally_independent(self, s1, s2, cond):
        """
        Description:
            - Determines whether the set of variables in s1 is independent of 
                all the variables in set s2, conditioning on the variables in 
                cond.

        Args:
            - s1: first set of variables
            - s2: second set of variables
            - cond: set of variables on which to condition
        """
        return self.graph.path(s1, s2, cond) == []
