import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_prob_dist(N):
    prob = np.random.randint(0,100, N)
    return prob / np.sum(prob)

def init_alias_table(prob_dist):
    N = len(prob_dist)
    norm_prob = prob_dist * N
    prob, alias = [0] * N, [0] * N
    small, large = [], []
    for i, p in enumerate(norm_prob):
        if p > 1.0:
            large.append(i)
        else:
            small.append(i)
    while small and large:
        small_index, large_index = small.pop(), large.pop()
        prob[small_index] = norm_prob[small_index]
        alias[small_index] = large_index
        norm_prob[large_index] = norm_prob[large_index] - (1.0 - norm_prob[small_index])
        if norm_prob[large_index] > 1.0:
            large.append(large_index)
        else:
            small.append(large_index)
    while large:
        prob[large.pop()] = 1
    while small:
        prob[small.pop()] = 1
    return prob, alias

def alias_sample(prob, alias):
    N = len(prob)
    i = int(np.random.random()*N)
    p = np.random.random()
    if p < prob[i]:
        return i
    else:
        return alias[i]

def simulate(N = 100, K = 1000):
    truth = get_prob_dist(N)
    prob, alias = init_alias_table(truth)
    sample = np.zeros(N)
    for _ in range(K):
        i = alias_sample(prob, alias)
        sample[i] += 1
    return truth, sample/np.sum(sample)

if __name__ == '__main__':
    truth, sample = simulate(N = 30, K = 100000)
    x = range(30)
    plt.figure()
    plt.plot(x, truth, '--o', label = 'truth')
    plt.plot(x, sample, '-x', label = 'sample')
    plt.legend()
    plt.show()



