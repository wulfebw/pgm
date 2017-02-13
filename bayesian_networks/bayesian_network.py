
import cpds

class BayesianNetwork(object):
    def __init__(self, graph, cpds):
        """
        Args:
            - graph: a networkx graph containing nodes for each variable
            - cpds: a list of cpds in topological order, one for each variable
                in the graph
        """
        self.graph = graph
        self.cpds = cpds

    def pdf(self, assignment):
        prob = 1.
        for cpd in self.cpds:
            prob *= cpd.pdf(assignment)
        return prob

    def logpdf(self, assignment):
        logprob = 0.
        for cpd in self.cpds:
            logprob += cpd.logpdf(assignment)
        return logprob