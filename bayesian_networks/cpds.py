
import numpy as np

class Distribution(object):
    def __init__(self):
        pass

class StaticDistribution(Distribution):
    def __init__(self, func):
        self.func = func
    def __call__(self, value):
        return self.func(value)

class CategoricalDistribution(Distribution):
    def __init__(self, table):
        """
        Args:
            - table: a vector of probabilities, where probability[i] equals the 
                probability of the ith value.
        """
        assert len(table) >= 2
        assert sum(table) == 1.0
        self.table = table

    def __call__(self, value):
        assert value <= len(self.table) - 1
        return self.table[value]

# conditional probability distribution base
class CPD(object):
    def __init__(self):
        pass

# probability distribution without parents
class StaticCPD(CPD):
    def __init__(self, variable, distribution):
        """
        Args:
            - variable: string name for this variable
            - distribution: function that takes an assignment of the variable 
                and outputs the probability of that assignment
        """
        self.variable = variable
        self.distribution = distribution

    def pdf(self, assignment):
        return self.distribution(assignment[self.variable])

    def logpdf(self, assignment):
        return np.log(self.pdf(assignment))

# discrete conditional probability distribution with parents
class CategoricalCPD(CPD):
    def __init__(self, variable, parents, num_categories, distributions):
        """
        Args:
            - variable: string, name of this node for this CPD
            - parents: parent variables (list of strings)
            - num_categories: the number of categories for each parent (list)
            - distributions: distribution for each unique parental instantiation
                collected into a dictionary s.t., 
                key = (parent_1_assignment, ..., parent_n_assignment), 
                and distributions[key] = distribution associated with that assignment
        """
        self.variable = variable
        self.parents = parents
        self.num_categories = num_categories
        self.distributions = distributions

    def pdf(self, assignment):
        """
        Args:
            - assignment: dictionary mapping a variable to a value, where the 
                value indicates which of the n possible instantiations of the 
                variable that variable is assuming.
        """ 
        # determine which conditional distribution to use based on parent values
        key = [assignment[p] for p in self.parents]
        key = tuple(key) if len(key) > 1 else key[0]
        return self.distributions[key](assignment[self.variable])

    def logpdf(self, assignment):
        return np.log(self.pdf(assignment))
