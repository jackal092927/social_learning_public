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
    p = 0.75 # Probability for 1 in y

    randomW = False
    save_plot = True
    save_results = True
    FJ_maxiters = 5
    FJ_iters = 0

    datasrc = None
    file_path = ""

    G=None

    # datasrc = "erdos_renyi"
    # G = select_model_or_dataset(datasource=datasrc, n=n, p=10./n, seed=SEED, directed=False)

    # datasrc = "barabasi_albert"
    # G = select_model_or_dataset(datasource=datasrc, n=n, m=5, seed=SEED)

    # datasrc = "watts_strogatz"
    # G = select_model_or_dataset(datasrc, n=n, k=5, p=0.25)

    # datasrc = "random_W"
    # W = select_model_or_dataset(datasource=datasrc, n=n, rho=rho)

    # real data
    datasrc = "real_dataset"
    data_filenames = ['chesapeake.mtx', 'bio-celegansneural.mtx', 'delaunay_n10.mtx', 'polblogs.mtx', 'delaunay_n11.mtx', 'delaunay_n12.mtx', 'delaunay_n13.mtx']
    file_path = 'polblogs.mtx'
    W = select_model_or_dataset(datasource=datasrc, FJ_maxiters=FJ_maxiters, file_path=file_path)

    n = W.shape[0]
    Y = initialize_y(n, k, p)
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
    DataFile: {file_path}
    Random W: {randomW}
    #nodes: {Z.shape[0]}
    #articles: {Z.shape[1]}
    W_Sparse: {np.sum(W==0)/(W.size)}
    Pr(y==1): {np.sum(Y==1)/(Y.size)}
    Count Z<0: {m}, {m/Z.size}
    FJ_iters: {FJ_iters}
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
    plt.title(f'Comparison of Greedy Algorithm and Random Picking on {datasrc}: {file_path}')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    # plt.text(0.95, 0.05, meta_info, 
    #         fontsize=8, verticalalignment='bottom', horizontalalignment='right', transform=plt.gca().transAxes)

    # Meta information as caption
    plt.figtext(0.5, -0.15, meta_info, wrap=True, horizontalalignment='left', fontsize=8)
    plt.legend()
    plt.grid(True)


    current_datetime = datetime.now()
    timestamp = current_datetime.strftime('%Y-%m-%d-%H-%M-%S')

    dir_str = "./output/"
    fstr = f"FJ3model_DataSrc_{datasrc}_{file_path}_RandomW{randomW}_Nodes{n}_Articles{Z.shape[1]}_Sparsity{np.sum(W==0)/(W.size)}_PrY1{np.sum(Y==1)/(Y.size)}_CountZNeg{m/Z.size}"


    if save_plot:
        plot_fstr = f"{dir_str}plot_{fstr}_{timestamp}.pdf"        
        plt.savefig(plot_fstr, format='pdf')
    print("Save to ==> ", plot_fstr)

    results_topkl = {
        "graph": datasrc,
        "filename": file_path,
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
        "FJ_iters": FJ_iters,
    }

    if save_results:
        result_fstr = f"{dir_str}result_{fstr}_{timestamp}.pkl"
        with open(result_fstr, "wb") as file:
            pickle.dump(results_topkl, file)
    
    

    print("Save to ==> ", result_fstr)




    plt.show()







if __name__ == '__main__':
    main()


