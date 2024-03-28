import numpy as np
import networkx as nx
import random
import copy
import matplotlib.pyplot as plt
import math

from scipy.io import mmread
import pickle as pkl
from datetime import datetime

from utils import *
from data_loader import *

n=128 
d = math.ceil(math.log2(n))

AlgorithmLabels = ['Random Selection', 'Degree Selection', 'Error Rate Selection','Degree*Error Selection', 'Greedy Algorithm Approximate','Greedy Algorithm']
AlgorithmMarkers = ['o', '+', 'x', 'p', '^', '*']


DATASrc_DIC = {
    "FB": "./output/result_exp8_degree__FixedOptModel_DataSrc_real_dataset_fb-pages-food.edges_RandomWFalse_Nodes620_Articles3_Sparsity0.7511238293444329_PrY10.5876344086021505_P_range=(0.3, 0.9)_CountZNeg0.27043010752688174_2024-03-25-00-08-26.pkl",
    "SOC" : "./output/result_exp8_degree__FixedOptModel_DataSrc_real_dataset_soc-hamsterster.edges_RandomWFalse_Nodes2426_Articles3_Sparsity0.660765926154486_PrY10.5948062654575432_P_range=(0.3, 0.9)_CountZNeg0.20967298708436383_2024-03-25-05-22-13.pkl",
    "WIKI": "./output/result_exp8_degree__FixedOptModel_DataSrc_real_dataset_soc-wiki-Vote.mtx_RandomWFalse_Nodes889_Articles3_Sparsity0.6593574003474537_PrY10.5841769778777652_P_range=(0.3, 0.9)_CountZNeg0.25984251968503935_2024-03-25-00-33-35.pkl",
    "WS":"./output/result_exp8_degree__FixedOptModel_DataSrc_watts_strogatz__RandomWFalse_Nodes128_Articles3_Sparsity0.73583984375_PrY10.5859375_P_range=(0.3, 0.9)_CountZNeg0.3020833333333333_2024-03-24-23-52-51.pkl",
    "randomW":"./output/result_exp8_degree__FixedOptModel_DataSrc_random_W__RandomWTrue_Nodes128_Articles3_Sparsity0.94842529296875_PrY10.5859375_P_range=(0.3, 0.9)_CountZNeg0.2942708333333333_2024-03-24-23-55-11.pkl",
    "BA": "./output/result_exp8_degree__FixedOptModel_DataSrc_barabasi_albert__RandomWFalse_Nodes128_Articles3_Sparsity0.374267578125_PrY10.5859375_P_range=(0.3, 0.9)_CountZNeg0.25_2024-03-24-23-49-15.pkl",
    "ER":"./output/result_exp8_degree__FixedOptModel_DataSrc_erdos_renyi__RandomWFalse_Nodes128_Articles5_Sparsity0.9786376953125_PrY10.625_P_range=(0.3, 0.9)_CountZNeg0.35_2024-03-24-23-38-47.pkl",
}

YLabel='Egalitarian Objective Value'
XLabel='k := #selected nodes for intervention'

save_plot = True
dir_str = "./output/exp8/"

def main():
    ### pkl filename
    datasrc = "WIKI"
    pkl_filename = DATASrc_DIC[datasrc]

    # fstr = f"{py_fname}_FixedOptModel_DataSrc_{datasrc}_{file_path}_RandomW{randomW}_Nodes{n}_Articles{Z.shape[1]}_Sparsity{np.sum(W==0)/(W.size)}_PrY1{np.sum(Y==1)/(Y.size)}_P_range={Prange}_CountZNeg{m/Z.size}"


    save_plot_fstr = f"exp8newplot_datasrc_{datasrc}"
    

    with open(pkl_filename, 'rb') as file:
        pkl_results = pickle.load(file)

    # Access the attributes of the loaded class instance
    graph = pkl_results["graph"]
    filename = pkl_results["filename"]
    matrixW = pkl_results["matrixW"]
    random_seed = pkl_results["random_seed"]
    nodesize_n = pkl_results["nodesize_n"]
    articlesize_k = pkl_results["articlesize_k"]
    sparsity_rho = pkl_results["sparsity_rho"]
    probability1 = pkl_results["probability1"]
    negative_size = pkl_results["negative_size"]
    total_iteration = pkl_results["total_iteration"]
    timestamp = pkl_results["timestamp"]
    FJ_iters = pkl_results["FJ_iters"]
    repeat_exp = pkl_results["repeat_exp"]
    repeatk = pkl_results["repeatk"]


    results = pkl_results["results"]
    print(results.shape)
    max_total_iterations = results.shape[1]

    mask = MASK
    algorithms_count = np.sum(np.array(MASK))
    objective_vals = results 
    objective_reps_vals = np.ones_like(objective_vals)    

    print("objective_vals.shape:\t" , objective_vals.shape, "max_total_iterations==",max_total_iterations) ### [AlgorithmCount X max_total_iterations]
    print(f"cover ratio @ step log(n)=={d}:\t", [f"{objective_vals[alg_id, d-1]:.2f}" for alg_id in range(algorithms_count)])

    # Plotting
    t_values = range(1, max_total_iterations + 1)
    # Annotating the points
    offset = 3/100.  # Adjust this offset to position your text
    x_highlight = d
    y_highlights = objective_vals[:, x_highlight - 1]
    plt.figure(figsize=(10, 6))
    for alg_id in range(algorithms_count):
        plt.plot(t_values, objective_vals[alg_id, :], label=AlgorithmLabels[alg_id], marker=AlgorithmMarkers[alg_id])
        y_highlight = y_highlights[alg_id] 
        plt.text(x_highlight, y_highlight + offset, f'{y_highlight:.2f}', fontsize=13, ha='center')

    # plt.plot(t_values, objective_greedy[:max_total_iterations], label='Greedy Algorithm', marker='x')
    # plt.plot(t_values, objective_greedy_appro[:max_total_iterations], label='Greedy Algorithm Approximate', marker='^')
    # plt.plot(t_values, objective_random[:max_total_iterations], label='Random Picking', marker='o')
    # plt.plot(t_values, objective_degree[:max_total_iterations], label='Degree Selection', marker='+')


    
    # y_greedy = objective_greedy[x_highlight - 1]  # Adjusting index for 0-based indexing
    # y_greedy_appro = objective_greedy_appro[x_highlight - 1]
    # y_random = objective_random[x_highlight - 1]

    

    # plt.scatter([x_highlight]*algorithms_count, [y_greedy, y_greedy_appro, y_random], color='red')
    plt.scatter([x_highlight]*algorithms_count, y_highlights, color='blue', marker='D')


    # plt.text(x_highlight, y_greedy + offset, f'{y_greedy:.2f}', fontsize=13, ha='center')
    # plt.text(x_highlight, y_greedy_appro + offset, f'{y_greedy_appro:.2f}', fontsize=13, ha='center')
    # plt.text(x_highlight, y_random + offset, f'{y_random:.2f}', fontsize=13, ha='center')


    plt.xlabel(XLabel)
    plt.ylabel(YLabel)
    plt.title(f'Comparison of Algorithms on Dataset={datasrc}')

    plt.legend()
    plt.grid(True)

    if save_plot:
        plot_fstr = f"{dir_str}plot_{save_plot_fstr}_{timestamp}.pdf"        
        plt.savefig(plot_fstr, format='pdf')
        print("Save to ==> ", plot_fstr)    

    plt.show()

    # thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    thresholds = [0.5, 0.75, 0.9, 0.95, 0.99]
    selection_up2thresholds = np.zeros([algorithms_count ,len(thresholds)])
    # step99s_greedy, step99s_greedy_appros, step99s_randoms, step99s_degrees  = [], [], [], []
    for id, threshold in enumerate(thresholds):
        selection_count = np.sum(objective_vals<=threshold, axis=1)
        selection_count += 1
        selection_up2thresholds[:, id] = selection_count
        print(f"Avg#Selection==>{threshold}:\t [{selection_count}],"  )


if __name__ == '__main__':
    main()