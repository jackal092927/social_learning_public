import numpy as np
import networkx as nx
import random
import copy
import matplotlib.pyplot as plt
import math

from scipy.io import mmread
import pickle
from datetime import datetime

from utils import *
from data_loader import *

# # Create a dictionary mapping keywords to functions or objects
# switch_dict = {
#     'erdos_renyi': erdos_renyi_graph,
#     'watts_strogatz': watts_strogatz_graph,
#     'barabasi_albert': barabasi_albert_graph,
#     'real_dataset': load_real_dataset,
# }





def main():
    SEED=13
    random.seed(SEED)
    np.random.seed(SEED)
    

    n = 1000
    k = 100
    rho = 0.1  # ratio of nonzeros in W
    p = 0.6 # Probability for 1 in y

    randomW = False
    save_plot = True
    save_results = True

    datasrc = "erdos_renyi"
    G = select_model_or_dataset(datasource=datasrc, n=n, p=10./n, seed=SEED, directed=False)

    # datasrc = "barabasi_albert"
    # G = select_model_or_dataset(datasource=datasrc, n=n, m=5, seed=SEED)

    # datasrc = "watts_strogatz"
    # G = select_model_or_dataset(datasrc, n=n, k=5, p=0.25)





    # # ER random graph
    # G = nx.erdos_renyi_graph(n, rho, seed=SEED, directed=False)

    # # BA random graph
    # G = nx.barabasi_albert_graph(n, 5, seed=None, initial_graph=None)

    # WS random graph
    # G = nx.watts_strogatz_graph(n=n, k=5, p=0.25)

    
    # SBM random graph
    # block_k = 3
    # sz = int(n/block_k)
    # sizes = [sz, sz, n-2*sz]
    # in_p = 0.25
    # out_p = 0.05
    # probs = [[in_p, out_p, out_p], [out_p, in_p, out_p], [out_p, out_p, in_p]]
    # # probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]
    # G = nx.stochastic_block_model(sizes, probs, seed=SEED)

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
    if randomW:
        W = initialize_W(n, rho)

    Y = initialize_y(n, k, p)
    A = nx.adjacency_matrix(G).toarray()
    W = A
    for i in range(1,3+1):
        W_ = construct_finiteFJmodelW_from_graph(t=i, W=A, y=Y)
        sparsity = np.sum(W_==0)/W_.size
        print(i, sparsity)
        # if sparsity > int(math.sqrt(n)) + 1:
        if sparsity > 0.1 and sparsity < 0.95:
            W = W_

    Z = W @ Y
    m = np.sum(Z < 0)

    # print("datasource:\t",  datasrc)
    # print("number of nodes:\t", n)
    # print("number of articles:\t", k)
    # print("sparsity factor of W:\t", rho)
    # print("probability of 1 in y:\t", p)

    # print("Number of negative elements in Z: ", m)

    # Meta information
    meta_info = f"""
    DataSource: {datasrc}
    Random W: {randomW}
    #nodes: {Z.shape[0]}
    #articles: {Z.shape[1]}
    W_Sparse: {np.sum(W==0)/(W.size)}
    Pr(y==1): {np.sum(Y==1)/(Y.size)}
    Count Z<0: {m}, {m/Z.size}
    """
    print(meta_info)

    # total_iterations = 2 * int(math.sqrt(n)) + 1
    max_iterations = 100
    early_stop = 0.99

    results, total_iterations =  experiment1(n, W, Y, Z, m=m, max_iterations=max_iterations, early_stop=early_stop)
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
    plt.title('Comparison of Greedy Algorithm and Random Picking on ' + datasrc)
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    # plt.text(0.95, 0.05, meta_info, 
    #         fontsize=8, verticalalignment='bottom', horizontalalignment='right', transform=plt.gca().transAxes)

    # Meta information as caption
    plt.figtext(0.5, -0.15, meta_info, wrap=True, horizontalalignment='left', fontsize=8)
    plt.legend()
    plt.grid(True)
    plt.show()


    current_datetime = datetime.now()
    timestamp = current_datetime.strftime('%Y-%m-%d-%H-%M-%S')

    dir_str = "./output/"
    fstr = f"FJ3model_DataSrc_{datasrc}_RandomW{randomW}_Nodes{n}_Articles{Z.shape[1]}_Sparsity{np.sum(W==0)/(W.size)}_PrY1{np.sum(Y==1)/(Y.size)}_CountZNeg{m/Z.size}"


    if save_plot:
        plot_fstr = f"plot_{fstr}.pdf"        
        plt.savefig(f"{dir_str}{plot_fstr}_{timestamp}", format='pdf')

    results = {
        "graph": datasrc,
        "matrixW": W, 
        "random_seed": SEED,
        "nodesize_n": n,
        "articlesize_k": k,
        "sparsity_rho": rho,  # Sparsity factor for W
        "probability1": p,  # Probability for 1 in y
        "negative_size": m,
        "results": results,
        "total_iteration": total_iterations,
        "timestamp": timestamp,
    }

    if save_results:
        result_fstr = f"result_{fstr}_{timestamp}.pkl"
        with open(f"{dir_str}{result_fstr}", "wb") as file:
            pickle.dump(results, file)
    
    

    print("Save to ==> ", f"{dir_str}{result_fstr}")




if __name__ == '__main__':
    main()


