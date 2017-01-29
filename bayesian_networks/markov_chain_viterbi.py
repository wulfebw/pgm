
import numpy as np

def markov_chain_viterbi(v, A, T):
    m = len(v)
    probs = np.empty((m, T))
    probs[:,0] = v
    best = np.empty((m, T))
    best[:, 0] = 0

    # compute for each state at each timestep the maximum probability 
    # of any sequence having led to that state
    for t in range(1, T):
        for s in range(m):
            best_prev_s, best_prob = 0, 0
            for prev_s in range(m):
                cur_prob = probs[prev_s, t - 1] * A[prev_s, s]
                if cur_prob > best_prob:
                    best_prob, best_prev_s = cur_prob, prev_s
            probs[s,t], best[s, t] = best_prob, best_prev_s
    max_prob = max(probs[:, -1])

    # go backward through the grid to retrieve the most likely sequence
    max_prob_seq = np.empty(T, dtype=int)
    max_prob_seq[-1] = np.argmax(probs[:, -1])
    for t in range(T - 1, 0, -1):
        max_prob_seq[t - 1] = best[max_prob_seq[t], t]

    return max_prob, max_prob_seq

if __name__ == '__main__':
    v = np.asarray([1/3., 1/3., 1/3.])
    A = np.asarray([[0, .2, .8], [.8, 0, .2], [.2, .8, 0]])
    T = 100
    prob, seq = markov_chain_viterbi(v, A, T)
    print('seq: {}\nprob: {}'.format(seq, prob))

"""
notes:
1. what's the difference between the forward algorithm and the viterbi algorithm?
    a. short answer: viterbi takes a max over the previous state probabilities whereas forward takes a sum
    b. longer answer: 
        - forward algorithm computes the probability of a certain state at each timestep given all the previous timesteps
            + does this by computing for each state at each timestep the probability of being in that state (i.e., of having transitioned to that state from the previous states)
        - viterbi algorithm computes the most likely state sequence (and its probability) at each timestep given all the previous timesteps
            + does this by computing for each state at each timestep the maximum possible probability of having transition to that state from a previous state
    c. some other confusing points:
        - neither requires "data" so long as there is not emission distribution
        - forward values will always be larger than viterbi values
        - viterbi is not useful for computing the full posterior values because these depend on future states (hence why backward algorithm is used in conjunction with forward to do that)
        - why don't you have to go backward in viterbi?
            + the answer is because you are computing different things
                * in viterbi, you are computing a value that only depends on past events (namely b/c we stop when we get to the end)
                    - i.e., at any given point in time, the future observations tell us nothing about the maximum probability with which each state could have been reached at a point in time
                * in forward-backward you compute values that depend both on the past
                and on the future
                    - i.e., the _probability_ of a state depends on the observations after it, and knowing those will change the probability
                * in other words, you can take a max in the viterbi case b/c the subproblems are fully solvable

            + true answer is that I'm not entirely sure, but I'll give it a shot
                * the reason is almost obviously that they compute different things, but how those differ to require different procedures in unclear
            + first note that you don't have to go backward in the forward algorithm
            + illuminating question is why _do_ you have to go backward in baum-welch?
                * answer to that is that you're computing a value over every possible transition - namely the posterior probability of being in a state at a time, and this requires information about moving both backward and forward
                * whereas with viterbi, 
        - so if you are asked for the maximum achievable probability of a sequence, then you would use the viterbi algorithm, not the forward algorithm
2. analysis
    a. running time:
        - T timesteps, each time computing for each state s of S a max over S states so O(TS^2)
    b. space:
        - requires O(T*S) grid
"""