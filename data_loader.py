import networkx as nx
from utils import *
from scipy.io import mmread

# Define your random graph models and real datasets
def erdos_renyi_graph(n=10, p=0.1, seed=0, directed=False):
    return nx.erdos_renyi_graph(n=n, p=p, seed=seed, directed=directed)

def watts_strogatz_graph(n=10, k=5, p=0.25):
    return nx.watts_strogatz_graph(n, k, p)

def barabasi_albert_graph(n=10, m=5, seed=0):
    return nx.barabasi_albert_graph(n=n, m=m, seed=seed)

def load_real_dataset(file_path):
    # Implement loading real dataset logic here
    data_dir = './dataset/'
    data_fname = data_dir + file_path
    a = mmread(data_fname)
    return nx.Graph(a)

def random_W(n, rho):
    return initialize_W(n, rho)

# Create a dictionary mapping keywords to functions or objects
switch_dict = {
    'erdos_renyi': erdos_renyi_graph,
    'watts_strogatz': watts_strogatz_graph,
    'barabasi_albert': barabasi_albert_graph,
    'real_dataset': load_real_dataset,
    'random_W': random_W,
}

# Function to select and use the appropriate model or dataset
def select_model_or_dataset(datasource, FJ_maxiters=10, *args, **kwargs):
    selected_function = switch_dict.get(datasource)
    if selected_function is None:
        raise ValueError(f"Invalid keyword: {datasource}")
    
    if datasource == "random_W":
        return selected_function(*args, **kwargs)
    else:
        G = selected_function(*args, **kwargs)
        W = selfadj_finiteFJmodelW(G, FJ_maxiters)
        return W
    

    # if datasource == 'real_dataset':
    #     return selected_function(*args)
    # else:
    #     return selected_function(*args, **kwargs)

def selfadj_finiteFJmodelW(G, FJ_maxiters=10):
    A = nx.adjacency_matrix(G).toarray()
    FJ_iters=0
    W=None
    
    for i in range(0,FJ_maxiters+1):
        W_ = construct_tstepFJmodelW_from_graph(t=i, W=A)
        sparsity = np.sum(W_==0)/W_.size
        print(i, sparsity)
            # if sparsity > int(math.sqrt(n)) + 1:
        if sparsity > 0.1:
            W = W_
            FJ_iters = i
        else: 
            break
    print("FJ_iters:", FJ_iters)
    return W
