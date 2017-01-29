
import bfs
import search_problem

def bfs_example():
    adj = {
        'A':['B','D'],
        'B':['A','C'],
        'C':['B','D'],
        'D':['A','C']
    }
    start = 'A'
    goals = set(['C'])
    ignores = set()
    problem = search_problem.UndirectedGraphSearchProblem(
        adj, start, goals, ignores)
    algorithm = bfs.BreadthFirstSearch()
    # valid path exists
    path = algorithm.solve(problem)
    print(path)

    # valid path does not exist
    ignores = set(('B','D'))
    problem = search_problem.UndirectedGraphSearchProblem(
        adj, start, goals, ignores)
    path = algorithm.solve(problem)
    print(path)

if __name__ == '__main__':
    bfs_example()