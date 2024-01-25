import numpy as np
import networkx as nx
import random
import copy
import matplotlib.pyplot as plt
import math

from utils import *
from scipy.io import mmread



def main():
    SEED=13
    random.seed(SEED)
    np.random.seed(SEED)
    

    n = 500
    k = 30
    rho = 0.1  # Sparsity factor for W
    p = 0.65  # Probability for 1 in y
    normalize = True





    # ER random graph
    # G = nx.erdos_renyi_graph(n, rho, seed=SEED, directed=False)

    # BA random graph
    G = nx.barabasi_albert_graph(n, 5, seed=None, initial_graph=None)

    # SBM random graph
    # block_k = 3
    # sz = int(n/block_k)
    # sizes = [sz, sz, n-2*sz]
    # in_p = 0.25
    # out_p = 0.05
    # probs = [[in_p, out_p, out_p], [out_p, in_p, out_p], [out_p, out_p, in_p]]
    # # probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]
    # G = nx.stochastic_block_model(sizes, probs, seed=SEED)

    # WS random graph
    # G = nx.watts_strogatz_graph(n=n, k=10, p=0.5)

    # GE random graph
    # G = nx.random_geometric_graph(n=n, radius=0.25)

    # real data
    # a = mmread('./dataset/delaunay_n10.mtx')
    # G = nx.Graph(a)
    # n = G.number_of_nodes()

    # print("number of nodes:\t", n)
    # print("number of articles:\t", k)
    # print("sparsity factor of W:\t", rho)
    # print("probability of 1 in y:\t", p)

    W = construct_W_from_graph(G)

    # randomW
    randomW = True
    if randomW:
        W = initialize_W(n, rho, normalize=normalize)

    Y = initialize_y(n, k, p)
    Z = W @ Y
    m = np.sum(Z < 0)

    print("number of nodes:\t", n)
    print("number of articles:\t", k)
    print("sparsity factor of W:\t", rho)
    print("probability of 1 in y:\t", p)

    print("Number of negative elements in Z: ", m)

    total_iterations = 2 * int(math.sqrt(n)) + 1

    # results =  experiment0(n, W, Y, Z, m, total_iterations)
    results, total_iterations =  experiment1(n, W, Y, Z, m)
    # print(results)
    objective_greedy, objective_greedy_appro, objective_random = results


    # Plotting
    t_values = range(1, total_iterations + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, objective_greedy, label='Greedy Algorithm', marker='x')
    plt.plot(t_values, objective_greedy_appro, label='Greedy Algorithm Approximate', marker='^')
    plt.plot(t_values, objective_random, label='Random Picking', marker='o')
    plt.xlabel('t (Subset Size)')
    plt.ylabel('Objective Value')
    plt.title('Comparison of Greedy Algorithm and Random Picking')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Meta information
    meta_info = f"""`
    Number of nodes: {n}
    Number of articles: {Y.shape[1]}
    Sparsity factor of W: {rho}
    Probability of 1 in y: {p}
    Number of negative vals: {m}
    Random W: {randomW}
    """
    # plt.text(0.95, 0.05, meta_info, 
    #         fontsize=8, verticalalignment='bottom', horizontalalignment='right', transform=plt.gca().transAxes)

    # Meta information as caption
    plt.figtext(0.5, -0.15, meta_info, wrap=True, horizontalalignment='left', fontsize=8)


    plt.legend()
    plt.grid(True)
    plt.show()




if __name__ == '__main__':
    main()


