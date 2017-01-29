class SearchProblem(object):
    # Return the start state.
    def start_state(self): raise NotImplementedError("Override me")

    # Return whether |state| is a goal state or not.
    def is_goal(self, state): raise NotImplementedError("Override me")

    # Return a list of (action, newState, cost) tuples corresponding to edges
    # coming out of |state|.
    def succ_and_cost(self, state): raise NotImplementedError("Override me")

"""
Examples:
"""

class DAGSearchProblem(SearchProblem):
    def __init__(self, adj, start): 
        self.adj = adj
        self.start = start
        self.goals = goals
    def start_state(self): return self.start
    def is_goal(self, state): return state in self.goals
    def succ_and_cost(self, state):
        assert state in self.adj
        cost = 1000 if (state == 3 or state == 5) else 1
        return [('right', ns, cost) for ns in self.adj[state]]

class UndirectedGraphSearchProblem(SearchProblem):
    def __init__(self, adj, start, goals, ignores):
        self.adj = adj
        self.start = start
        self.goals = goals
        self.ignores = ignores
    def start_state(self): return self.start
    def is_goal(self, s): return s in self.goals
    def succ_and_cost(self, s):
        return [(n, 1) for n in self.adj[s] if n not in self.ignores]
