

import collections
import copy
import itertools

def get_key(variables, i, j):
    return tuple(sorted((variables[i], variables[j])))

def get_witnesses(variables, i, j):
    temp = set(copy.deepcopy(variables))
    temp.discard(variables[i])
    temp.discard(variables[j])
    temp.add('.')
    for length in range(1, len(temp) + 1):
        for combination in itertools.combinations(temp, r=length):
            yield set(combination)

def are_independent(variables, i, j, witness, indps):
    key = get_key(variables, i, j)
    for indp in indps[key]:
        if len(witness.symmetric_difference(indp)) == 0:
            return True
    return False

def build_skeleton(variables, indps):
    # build fully connected undirected graph
    num_vars = len(variables)
    adj = collections.defaultdict(set)
    for i in range(num_vars):
        for j in range(num_vars):
            if i == j: continue
            adj[variables[i]].add(variables[j])

    # initialize the witnesses
    witnesses = collections.defaultdict(set)

    # for each pair of variables, determine if they can be separated
    for i in range(num_vars):
        for j in range(num_vars):
            if i == j: continue

            # go through every possible witness
            # if that witness separates the variables, then remove the edge
            for witness in get_witnesses(variables, i, j):
                if are_independent(variables, i, j, witness, indps):
                    adj[variables[i]].discard(variables[j])
                    adj[variables[j]].discard(variables[i])
                    key = get_key(variables, i, j)
                    witnesses[key].add(tuple(witness))

    # return the skeleton and witnesses
    return adj, witnesses

def find_potential_immoralities(skeleton):
    # finds all of the X-Z-Y triples s.t. exists edge X-Z-Y, but not X-Y
    potential_immoralities = set()
    for x in skeleton.keys():
        for z in skeleton[x]:
            for y in skeleton[z]:
                if y != x and x not in skeleton[y]:
                    first, second = sorted([x,y])
                    potential_immoralities.add((first,z,second))
    return potential_immoralities

def find_true_immoralities(potential_immoralities, witnesses):
    # true immoralities are those s.t. z is not a witness to the independence 
    # of x and y given z (i.e., conditional on z, x and y are still dependent)
    immoralities = set()
    for potential_immorality in potential_immoralities:
        x, z, y = potential_immorality
        key = tuple(sorted((x, y)))
        valid = True
        for witness in witnesses[key]:
            if z in witness:
                valid = False
                break
        if valid:
            immoralities.add((x,z,y))
    return immoralities

def run_p_map():
    
    variables = ['A', 'B', 'C', 'D']
    indps = collections.defaultdict(list)
    # add independencies
    indps[('A','C')] += [set(v) for v in ['B', ['B','D']]]
    indps[('A','D')] += [set(v) for v in ['.', 'B', ['B','C']]]
    indps[('B','D')] += [set(v) for v in ['.', 'A']]

    skeleton, witnesses = build_skeleton(variables, indps)
    print('skeleton: {}'.format(skeleton))
    print('witnesses: {}'.format(witnesses))

    potential_immoralities = find_potential_immoralities(skeleton)
    true_immoralities = find_true_immoralities(potential_immoralities, witnesses)
    print('potential_immoralities: {}'.format(potential_immoralities))
    print('true immoralities: {}'.format(true_immoralities))

if __name__ == '__main__':
    run_p_map()