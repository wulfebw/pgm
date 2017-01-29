
import collections

import search_algorithm

Vertex = collections.namedtuple('Vertex', ['value', 'parent'])

class BreadthFirstSearch(search_algorithm.SearchAlgorithm):

    def solve(self, problem):
        v = set()
        q = collections.deque()

        s = problem.start_state()
        s = Vertex(s, None)
        q.appendleft(s)
        while len(q) > 0:
            s = q.pop()
            if problem.is_goal(s.value):
                # backtrack path
                parent, path = s.parent, collections.deque()
                while parent is not None:
                    path.appendleft(parent.value)
                    parent = parent.parent
                return list(path)
            else:
                # ignore cost
                for n, _ in problem.succ_and_cost(s.value):
                    n = Vertex(n, s)
                    q.appendleft(n)
        # did not reach goal, return empty list to indicate no path
        return []

