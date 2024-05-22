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
from utils import plot_results


# file_paths = []
# file_paths.append('./output/exp7/result_exp5_avg_performance_FixedOptModel_DataSrc_erdos_renyi__RandomWFalse_Nodes1024_Articles10_Sparsity0.8503646850585938_PrY10.6064453125_CountZNeg0.23251953125_2024-02-01-00-44-13.pkl')
# file_paths.append('./output/exp7/result_exp7_avg__FixedOptModel_DataSrc_barabasi_albert__RandomWFalse_Nodes1024_Articles10_Sparsity0.1747608184814453_PrY10.6064453125_CountZNeg0.105078125_2024-02-01-01-01-13.pkl')
# file_paths.append('./output/exp7/result_exp7_avg__FixedOptModel_DataSrc_watts_strogatz__RandomWFalse_Nodes1024_Articles10_Sparsity0.9631137847900391_PrY10.6064453125_P_range=(0.3, 0.9)_CountZNeg0.27705078125_2024-02-01-08-35-36.pkl')
# file_paths.append('./output/exp7/result_exp7_avg__FixedOptModel_DataSrc_real_dataset_bio-celegansneural.mtx_RandomWFalse_Nodes297_Articles10_Sparsity0.44970467866090763_PrY10.602020202020202_P_range=(0.3, 0.9)_CountZNeg0.11414141414141414_2024-02-01-13-29-43.pkl')
# file_paths.append('./output/exp7/result_exp7_avg__FixedOptModel_DataSrc_real_dataset_chesapeake.mtx_RandomWFalse_Nodes39_Articles10_Sparsity0.7508218277449047_PrY10.6102564102564103_P_range=(0.3, 0.9)_CountZNeg0.23076923076923078_2024-02-01-13-27-22.pkl')
# file_paths.append('./output/exp7/result_exp7_avg__FixedOptModel_DataSrc_real_dataset_fb-pages-food.edges_RandomWFalse_Nodes620_Articles10_Sparsity0.7511238293444329_PrY10.6011290322580645_P_range=(0.3, 0.9)_CountZNeg0.2567741935483871_2024-02-01-14-52-22.pkl')
# file_paths.append('./output/exp7/result_exp7_avg__FixedOptModel_DataSrc_real_dataset_soc-wiki-Vote.mtx_RandomWFalse_Nodes889_Articles10_Sparsity0.6593574003474537_PrY10.5974128233970754_P_range=(0.3, 0.9)_CountZNeg0.2559055118110236_2024-02-01-14-15-17.pkl')

DATASrc_DIC = {
    "FB": "./output/result_exp8_degree__FixedOptModel_DataSrc_real_dataset_fb-pages-food.edges_RandomWFalse_Nodes620_Articles3_Sparsity0.7511238293444329_PrY10.5876344086021505_P_range=(0.3, 0.9)_CountZNeg0.27043010752688174_2024-03-25-00-08-26.pkl",
    "SOC" : "./output/result_exp8_degree__FixedOptModel_DataSrc_real_dataset_soc-hamsterster.edges_RandomWFalse_Nodes2426_Articles3_Sparsity0.660765926154486_PrY10.5948062654575432_P_range=(0.3, 0.9)_CountZNeg0.20967298708436383_2024-03-25-05-22-13.pkl",
    "CSPK": "./output/result_exp8_degree__FixedOptModel_DataSrc_real_dataset_chesapeake.mtx_RandomWFalse_Nodes39_Articles3_Sparsity0.7508218277449047_PrY10.5128205128205128_P_range=(0.3, 0.9)_CountZNeg0.49572649572649574_2024-05-21-20-48-24.pkl",
    'BIO': "./output/result_exp8_degree__FixedOptModel_DataSrc_real_dataset_bio-celegansneural.mtx_RandomWFalse_Nodes297_Articles3_Sparsity0.44970467866090763_PrY10.5914702581369248_P_range=(0.3, 0.9)_CountZNeg0.2244668911335578_2024-05-21-20-50-49.pkl",
    "WIKI": "./output/result_exp8_degree__FixedOptModel_DataSrc_real_dataset_soc-wiki-Vote.mtx_RandomWFalse_Nodes889_Articles3_Sparsity0.6593574003474537_PrY10.5841769778777652_P_range=(0.3, 0.9)_CountZNeg0.25984251968503935_2024-03-25-00-33-35.pkl",
    "WS":"./output/result_exp8_degree__FixedOptModel_DataSrc_watts_strogatz__RandomWFalse_Nodes128_Articles3_Sparsity0.73583984375_PrY10.5859375_P_range=(0.3, 0.9)_CountZNeg0.3020833333333333_2024-03-24-23-52-51.pkl",
    "randomW":"./output/result_exp8_degree__FixedOptModel_DataSrc_random_W__RandomWTrue_Nodes128_Articles3_Sparsity0.94842529296875_PrY10.5859375_P_range=(0.3, 0.9)_CountZNeg0.2942708333333333_2024-03-24-23-55-11.pkl",
    "PA": "./output/result_exp8_degree__FixedOptModel_DataSrc_barabasi_albert__RandomWFalse_Nodes128_Articles3_Sparsity0.374267578125_PrY10.5859375_P_range=(0.3, 0.9)_CountZNeg0.25_2024-03-24-23-49-15.pkl",
    "ER":"./output/result_exp8_degree__FixedOptModel_DataSrc_erdos_renyi__RandomWFalse_Nodes128_Articles5_Sparsity0.9786376953125_PrY10.625_P_range=(0.3, 0.9)_CountZNeg0.35_2024-03-24-23-38-47.pkl",
}


dataset_names=["ER", "PA", "WS", 'randomW', 'BIO', 'CSPK', 'FB', 'WIKI' ]


threshold_i = 4 # 0.9
metrics = []
for dataset in dataset_names:

    # Open the file in binary read mode
    file_path = DATASrc_DIC[dataset]
    with open(file_path, 'rb') as file:
        # Load the data from the file
        data = pickle.load(file)

    results = data['results']
    objective_greedy, objective_greedy_appro, objective_random = results

    # thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    thresholds = 0.9
    step99s_greedy, step99s_greedy_appros, step99s_randoms  = [], [], []
    for threshold in thresholds:
        step99s_greedy.append(np.sum(np.array(objective_greedy)<=threshold))
        step99s_greedy_appros.append(np.sum(np.array(objective_greedy_appro)<=threshold))
        step99s_randoms.append(np.sum(np.array(objective_random)<=threshold))


    for i, threshold in enumerate(thresholds):
        print(f"#steps==>{threshold}:\t [{step99s_greedy[i]}, {step99s_greedy_appros[i]}, {step99s_randoms[i]}]"  )
        # print(f"step99s:\t", step99s)
    
    # ds_names = ['barabasi_albert', 'watts_strogatz',  'erdos_renyi']
    
    metrics.append([step99s_greedy[threshold_i]+1, step99s_greedy_appros[threshold_i]+1, step99s_randoms[threshold_i]+1])
    print(metrics)



plot_results8(metrics, dataset_names=dataset_names, ylabel="Steps", title="Comparison of Steps for Cover Ratio > 90%", xlabel="Datasets")