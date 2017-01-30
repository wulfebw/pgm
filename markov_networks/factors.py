
import numpy as np

class Factor(object):
    """
    Description:
        - Factor represents a potential function over a set of variables. this is a base class.
    """

    def __init__(self, clique):
        self.clique = clique

    def affinity(self, assignment):
        raise(NotImplementedError("affinity not implemented for base factor."))

class TabularFactor(Factor):
    """
    Description:
        - A factor that represents the affinities as a table of values.
    """
    def __init__(self, clique, table):
        super(TabularFactor, self).__init__(clique)
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
        if not key: # empty tuple indicates variables not given values 
            return 1
        else:
            return self.table[key]

def gaussian_pdf(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(
        - (x - mu) ** 2 / (2 * sigma ** 2))

class LinearGaussianFactor(Factor):
    """
    Description: (read this to the end because changes mind part way through)
        - A linear gaussian factor is one that parameterizes it's potential 
        function as a normal distribution, the mean of which is a linear 
        combination of the values of the variables in the factor, and the 
        variance of which is a constant (here at least, not sure this is 
        necessarily the case). Also, I'm not entirely sure this is a thing 
        for markov networks. Clearly it is for bayesian networks, but I'm
        not sure it makes sense in the markov network case because the CPD does 
        not have to be a probability distribution, so why make this gaussian? 
        Having said that, it seems plausible that you might want the relationship
        between sets of variables to be gaussian, so just go with it. The
        problem is that, it doesn't make sense - you have to return a value from
        the affinity function, but when you specify the mean as a function of 
        the nodes in the clique, you are specifying a probability distribution 
        rather than a scalar. The solution then is to define the mean and sigma
        of the distribution beforehand, and then have the value of x be determined 
        by a weighted sum of the values of the nodes in the clique. Then you can
        give a value reflecting the affinity for that setting of the variables.

    """
    def __init__(self, clique, mean=0, sigma=1, weights=None):
        super(LinearGaussianFactor, self).__init__(clique)
        self.weights = weights
        self.mean = mean
        self.sigma = sigma

    def affinity(self, assignment):
        """
        Description:
            - Return the affinity associated with an assignment, or 1 if the 
                assignment contains no variables associated with this factor.

        Args:
            - assignment: dict mapping variables to values
        """
        values = [assignment[var] for var in self.clique]
        x = (np.dot(values, self.weights) if self.weights is not None 
            else np.sum(values))
        return gaussian_pdf(x, self.mean, self.sigma)
