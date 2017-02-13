
import numpy as np

def marginalize(table, i):
    return table.sum(axis=i)

def marginalize_other(table, i, j):
    axes = list(range(len(table.shape)))
    axes.remove(i)
    axes.remove(j)
    axes = set(axes)
    marginal_axes = set(range(len(table.shape))).symmetric_difference(axes)
    return table.sum(axis=tuple(marginal_axes))

def cpd(table, i, given=None):
    if given is None:
        axes = list(range(len(table.shape)))
        axes.remove(i)
        return table.sum(axis=tuple(axes))
    else:
        

if __name__ == '__main__':
    # (a)
    # a,b,c,d
    table = np.zeros((2,2,2,2))
    table[0,0,0,0] = 1./8
    table[0,0,1,1] = 1./8
    table[1,0,1,0] = 1./4
    table[1,1,0,1] = 1./4
    table[0,1,1,1] = 1./4
    marginalized_a_b = marginalize_other(table, 2, 3)
    print(marginalized_a_b)
    marginalized_a_d = marginalize_other(table, 1, 3)
    print(marginalized_a_d)

    # (b)
    d_cpd = cpd(table, 3)
    print('d_cpd: {}'.format(d_cpd))
    c_cpd = cpd(table, 2)
    print('c_cpd: {}'.format(c_cpd))
    b_cpd = cpd(table, 1, (0,3))
