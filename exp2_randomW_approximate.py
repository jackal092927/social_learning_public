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
    k = 10
    rho = 0.1  # Sparsity factor for W
    p = 0.66  # Probability for 1 in y





    # ER random graph
    # G = nx.erdos_renyi_graph(n, rho, seed=SEED, directed=False)

    # BA random graph
    # G = nx.barabasi_albert_graph(n, 30, seed=None, initial_graph=None)

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

    # W = construct_W_from_graph(G)

    # randomW
    W = initialize_W(n, rho)

    Y = initialize_y(n, k, p)
    Z = W @ Y
    m = np.sum(Z < 0)

    print("number of nodes:\t", n)
    print("number of articles:\t", k)
    print("sparsity factor of W:\t", rho)
    print("probability of 1 in y:\t", p)

    print("Number of negative elements in Z: ", m)

    total_iterations = 2 * int(math.sqrt(n)) + 1

    # Calculate objective value using greedy algorithm
    # t_values = range(1, int(math.sqrt(n)) + 1)
    # S_greedy = greedy_approximation(W, y)
    # objective_greedy = calculate_objective(W, y, S_greedy)



    # objective_greedy = [calculate_objective(Z, W, Y, greedy_approximation(W, Y,max_iter=t))/m for t in t_values]
    # print(objective_greedy)

    # random_repeat = 10

    # # Calculate objective values for random subsets
    # objective_random = [ np.mean([ calculate_objective(Z, W, Y, pick_k_set(n, t))/m for _ in range(random_repeat)]) for t in t_values]

    # print(objective_random)

    # For the greedy_approximation
    # start_time_greedy = time.time()
    
    objective_greedy = []
    # total_iterations_greedy = total_iterations
    # greedy_selection = approximate_greedy_approximation(W, Y, max_iter=total_iterations)
    greedy_selection = greedy_approximation(W, Y, max_iter=total_iterations)
    for t in range(total_iterations):
        objective_value = calculate_objective(Z, W, Y, greedy_selection[:t+1]) / m
        objective_greedy.append(objective_value)

        # # Progress and time tracking
        # if t % (total_iterations_greedy // 10) == 0 or i == total_iterations_greedy - 1:
        #     percent_complete = (t + 1) / total_iterations_greedy * 100
        #     elapsed_time = time.time() - start_time_greedy
        #     print(f"Greedy Progress: {percent_complete:.2f}%, Time elapsed: {elapsed_time:.2f} seconds")

    print("objective_greedy:", objective_greedy)
        
    objective_greedy_appro = []
    greedy_selection_appro = approximate_greedy_approximation(W, Y, max_iter=total_iterations)
    for t in range(total_iterations):
        objective_value = calculate_objective(Z, W, Y, greedy_selection_appro[:t+1]) / m
        objective_greedy_appro.append(objective_value)

    print("objective_greedy_appro:", objective_greedy_appro)



    # For the second for-loop
    random_repeat = 10
    # start_time_random = time.time()
    objective_random = []
    # total_iterations_random = len(t_values)

    # for i, t in enumerate(t_values):
    random_selection = pick_k_set(n, t)
    for t in range(total_iterations):
        random_values = [calculate_objective(Z, W, Y, random_selection[:t+1]) / m for _ in range(random_repeat)]
        mean_objective_value = np.mean(random_values)
        objective_random.append(mean_objective_value)

        # # Progress and time tracking
        # if i % (total_iterations_random // 10) == 0 or i == total_iterations_random - 1:
        #     percent_complete = (i + 1) / total_iterations_random * 100
        #     elapsed_time = time.time() - start_time_random
        #     print(f"Random Progress: {percent_complete:.2f}%, Time elapsed: {elapsed_time:.2f} seconds")

    print("objective_random:", objective_random)



    # Plotting
    t_values = range(1, total_iterations + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, objective_random, label='Random Picking', marker='o')
    plt.plot(t_values, objective_greedy, label='Greedy Algorithm', marker='x')
    plt.plot(t_values, objective_greedy_appro, label='Greedy Algorithm Approximate', marker='^')
    plt.xlabel('t (Subset Size)')
    plt.ylabel('Objective Value')
    plt.title('Comparison of Greedy Algorithm and Random Picking')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    main()


