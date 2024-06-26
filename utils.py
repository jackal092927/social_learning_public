import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
plt.rcParams['savefig.format'] = 'pdf' 

from scipy.io import mmread
import pickle
import networkx as nx
import math



def initialize_W(n, rho, normalize=True, seed=0):
    """
    Initialize a random nonnegative n x n matrix with sparsity controlled by rho.

    :param n: Size of the matrix (n x n)
    :param rho: Sparsity factor (between 0 and 1), with rho=1 meaning a fully dense matrix
    :return: Randomly generated nonnegative matrix W
    """
    np.random.seed(seed=seed)

    # Create a random matrix with values between 0 and 1
    W = np.random.rand(n, n)

    # Make the matrix sparse by setting a fraction of elements to zero
    if rho < 1:
        mask = np.random.rand(n, n) > rho
        W[mask] = 0
        
    if normalize:
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums==0]=1e-10
        W = W / row_sums

    return W

def initialize_y(n, k, p):
    """
    Initialize an n x k binary matrix y, where each column is a random vector 
    from a Bernoulli distribution with probability p for 1.

    :param n: Number of rows
    :param k: Number of columns
    :param p: Probability of 1 in the Bernoulli distribution
    :return: Randomly generated binary matrix y
    """
    ### Create a binary {+1, -1} matrix based on Bernoulli distribution
    y_binary = np.random.binomial(1, p, (n, k))
    y = 2 * y_binary - 1 ## {0,1} ==> {-1, +1}

    return y

def initialize_y_with_randomP(n, k, p=None, Prange=(0.5, 1.), seed=0):
    """
    Initialize an n x k binary {-1,+1} matrix y, where each row is a random vector 
    sampled iid from a Bernoulli distribution with probability Pr(=1) 
    sampled from some distribution, by default uniform distribution in [0,1].

    :param n: Number of rows
    :param k: Number of columns
    :param p: Probability of 1 in the Bernoulli distribution
    :return: Randomly generated binary matrix y
    """
    np.random.seed(seed=seed)
    if p is None:
        p = np.random.uniform(low=Prange[0], high=Prange[1], size=n)

    # Initialize the matrix
    Y = np.zeros((n, k))

    for i in range(n):
        # Sample p_i for each row from a uniform distribution [0,1]
        # p_i = np.random.uniform()

        # Generate the i-th row with entries sampled from a Bernoulli distribution
        # with probability p_i, and then map 0s to -1s and 1s to +1s
        Y[i, :] = 2 * np.random.binomial(1, p[i], k) - 1
    
    # print(Y.shape)
    # print(np.sum(Y==1, axis=1)/(Y.shape[1]))
    # print(np.sum(Y==1, axis=0)/(Y.shape[0]))

    return Y




def calculate_objective(Z, W, Y, S):
    # objective_value = 0
    # for i in range(Y.shape[0]): # Loop over rows of y
    #     for a in range(Y.shape[1]):  # Loop over columns of y
    #         if Z[i,a] < 0 and sum(W[i,t] * (Y[t, a]-1) for t in S) < 0: 
    #             objective_value += 1


    # return objective_value

    n, k = Y.shape
    total_sum = 0

    for i in range(n):
        for a in range(k):
            # condition1 = np.dot(W[i, :], Y[:, a]) < 0
            condition1 = Z[i,a]< 0

            condition2 = any((Y[t, a] < 0 and W[i, t] > 0) for t in S)

            if condition1 and condition2:
                total_sum += 1

    return total_sum

def marginal_gain(Z, W, Y, S, t):
    """
    Calculate the marginal gain of adding element t to set S.

    :param Z: Matrix Z = WY
    :param W: Matrix W
    :param Y: Matrix Y
    :param S: Set S
    :param i: Element i
    :return: Marginal gain of adding element t to set S
    """ 
    n, k = Y.shape
    total_sum = 0

    for i in range(n):
        for a in range(k):

            # condition1 = np.dot(W[i, :], Y[:, a]) < 0
            condition1 = Z[i, a] < 0
            condition2 = W[i, t] > 0 and Y[t, a] < 0
            condition3 = all(( W[i, s] == 0 or Y[s, a] > 0) for s in S)

            if condition1 and condition2 and condition3:
                total_sum += 1

    return total_sum

def approximate_marginal_gain(Z, W, Y, S, t):

    n, k = Y.shape
    if k >= 2:
        Y=np.mean(Y<0, axis=1, keepdims=True)
    total_sum = 0.
    idx = []
    vals = []

    # E = W @ (1-2*Y)
    for i in range(n):
        sum1 = np.sum([ W[i,j] for j in S])
        # print(sum1)
        sum2 = np.sum( [ W[i,j] * (1. - 2.*Y[j,0] )- W[i,t] for j in range(n) if j not in S and j != t ] ) 
        # print(sum2)
        phi = sum1 + sum2
        # print(phi)
        # sum2 = np.sum( [ W[i,j] * (1. - 2.*Y[j,0] ) for j in range(n) if j not in S and j != t ] ) 
        # phi = sum1 + sum2 - W[i,t] 

        # [0.] [0.14303235] [-0.00993206] [0.01655343]
        if phi >= 0: # phi[14]=-4.068159193049652
            continue
        
        # idx.append(i)
        # vals.append(phi)
        pi = 1.
        for j in S:
            if W[i,j] == 0:
                continue
            # pi = pi * 1. - Y[j,0]
            pi = pi * (1. - Y[j,0])

        # total_sum += pi * Y[i,0]
        total_sum += pi * Y[t,0]

        
            # # condition1 = np.dot(W[i, :], Y[:, a]) < 0
            # condition1 = Z[i, a] < 0
            # condition2 = W[i, t] > 0 and Y[t, a] < 0
            # condition3 = all(( W[i, s] == 0 or Y[s, a] > 0) for s in S)

            # if condition1 and condition2 and condition3:
            #     total_sum += 1
    # print(total_sum)
    # print(idx)
    # print(vals)
    return total_sum

def approximate_marginal_gain_opt(Z, W, Y, S, t):

    n, k = Y.shape
    if k >= 2:
        Y=np.mean(Y<0, axis=1, keepdims=True)
    total_sum = 0.

    # E = W @ (1-2*Y)
    for i in range(n):
        sum1 = np.sum([ W[i,j] for j in S])
        sum2 = np.sum( [ W[i,j] * (1. - 2.*Y[j,0] ) for j in range(n) if j not in S and j != t ] ) 
        phi = sum1 + sum2 - W[i,t] 
        # if i==0:
        #     if phi >= 0:
        #         print(sum1, sum2, phi)
        if phi >= 0:
            continue
        
        pi = 1.
        for j in S:
            if W[i,j] == 0:
                continue
            pi = pi * 1. - Y[j,0]

        total_sum += pi * Y[i,0]

        
            # # condition1 = np.dot(W[i, :], Y[:, a]) < 0
            # condition1 = Z[i, a] < 0
            # condition2 = W[i, t] > 0 and Y[t, a] < 0
            # condition3 = all(( W[i, s] == 0 or Y[s, a] > 0) for s in S)

            # if condition1 and condition2 and condition3:
            #     total_sum += 1
    print(total_sum)
    return total_sum

def approximate_greedy_approximation(W, Y, max_iter=100):
    if Y.shape[1] >= 2:
        Y = np.mean(Y<0, axis=1, keepdims=True)
    # print(Y.shape)
    Z = W @ Y
    n = W.shape[0]
    S = set()
    S_list = []
    best_increase = 0
    selected_i = None
    gain_zero = False

    for t in range(max_iter):
        best_increase = 0
        selected_i = None

        for i in range(n):
            if i in S:
                continue
            if gain_zero:
                if selected_i is None:
                    selected_i = i
                    S.add(i)
                    S_list.append(i)
                break   

            # S.add(i)
            # increase = calculate_objective(Z, W, Y, S) - calculate_objective(Z, W, Y, S - {i})
            # if increase != marginal_gain(Z, W, Y, S - {i}, i):
            #     print("Error")
            # if increase > best_increase:
            #     best_increase = increase
            #     selected_i = i
            # S.remove(i)

            gain = approximate_marginal_gain(Z, W, Y, S, i)
            # if gain == 0: # first time gain==0
            #     gain_zero = True
            #     print("Iteration:", t, "\t gain==0")
            # if selected_i is None:
            #     selected_i = i
            if gain > best_increase:
                best_increase = gain
                selected_i = i
        
        # if best_increase == 0:
        #     print("Iteration:", t, "\t gain==0")

        if selected_i is not None:
            S.add(selected_i)
            S_list.append(selected_i)
            # print("Selected element:", selected_i, "with marginal gain:", best_increase)
        else:
            gain_zero = True
            print("Iteration:", t, "\t gain==0")
            # break  # Break if no improvement
            break

    return S_list

def approximate_greedy_approximation_opt(W, Y, max_iter=100):
    if Y.shape[1] >= 2:
        Y = np.mean(Y<0, axis=1, keepdims=True)
    # print(Y.shape)
    Z = W @ Y
    n = W.shape[0]
    S = set()
    S_list = []
    best_increase = 0
    selected_i = None
    gain_zero = False

    WE = W @ (1-2*Y)
    # print("WE.shape:", WE.shape)
    WS = np.zeros([n,1])
    # WE_S = np.sum(WE, axis=1, keepdims=True)
    # WE_S = 
    EW = (np.ones([n,1])@((1-Y).T) * (W!=0))+(W==0)
    W2Y = W * (np.ones([n,1])@((1-2*Y).T))
    # print("EW.shape:", EW.shape)
    # assert ((EW==1)==(W==0)).all(), "EW error"  
    E_S = np.ones([n,1])
    flag = WS + WE
    flags =[]

    for t in range(max_iter):
        best_increase = 0
        selected_i = None

        for i in range(n):
            if i in S:
                continue
            if gain_zero:
                if selected_i is None:
                    selected_i = i
                    S.add(i)
                    S_list.append(i)
                break   
            

            # T1 = WS[i]
            # T2 = WE_S[i] 
            # T3 = E_S[i]

            # flag = WS + WE - W2Y[:, i:i+1] - (n-t-1)*W[:,i:i+1]  # in fact this performs better
            flag = WS + WE - W2Y[:, i:i+1] - W[:,i:i+1]
            # print(WS[14], WE_S[14], W2Y[14, i:i+1], W[14,i:i+1])
            gain = np.sum((flag < 0) * E_S * Y[i,0], axis=0)[0]
            # idx = np.where(flag<0)
            # [ 14,  16,  26,  29,  32,  50,  58,  60,  71,  83,  86,  87,  92,
            # 98, 117, 142, 143, 145, 146, 170, 185, 190, 191, 194, 208, 216, 219, 231, 243, 251, 254]
            # print(idx)
            # if flag > 0:
            #     continue
            # gain_opt = Y[t,0] * E_S[i]

            # gain = approximate_marginal_gain(Z, W, Y, S, i)
            # [14, 16, 26, 29, 32, 50, 58, 60, 71, 83, 86, 87, 92, 
            # 98, 117, 142, 143, 145, 146, 170, 185, 190, 191, 194, 208, 216, 219, 231, 243, 251, 254]

            # assert math.isclose(gain, gain_opt, rel_tol=0.15), f"gain error: it={t}, i={i}, gain={gain}, gain_opt={gain_opt}"
            if gain > best_increase:
                best_increase = gain
                selected_i = i
        
        # if best_increase == 0:
        #     print("Iteration:", t, "\t gain==0")

        if selected_i is not None:
            S.add(selected_i)
            S_list.append(selected_i)
            # print("Selected element:", selected_i, "with marginal gain:", best_increase)
        else:
            gain_zero = True
            print("Iteration:", t, "\t gain==0")
            # break  # Break if no improvement
            break
        
        WS += W[:, selected_i:selected_i+1]
        WE -= - W2Y[:, selected_i:selected_i+1]
        E_S = E_S * EW[:, selected_i:selected_i+1]



    return S_list

def greedy_approximation(W, Y, max_iter=10):
    Z = W @ Y
    n = W.shape[0]
    S = set()
    S_list = []
    best_increase = 0
    selected_i = None

    for _ in range(max_iter):
        best_increase = 0
        selected_i = None

        for i in range(n):
            if i in S:
                continue

            # S.add(i)
            # increase = calculate_objective(Z, W, Y, S) - calculate_objective(Z, W, Y, S - {i})
            # if increase != marginal_gain(Z, W, Y, S - {i}, i):
            #     print("Error")
            # if increase > best_increase:
            #     best_increase = increase
            #     selected_i = i
            # S.remove(i)

            gain = marginal_gain(Z, W, Y, S, i)
            if gain > best_increase:
                best_increase = gain
                selected_i = i

        if selected_i is not None:
            S.add(selected_i)
            S_list.append(selected_i)
            # print("Selected element:", selected_i, "with marginal gain:", best_increase)
        else:
            break  # Break if no improvement
    
    assert len(S_list) == len(S), "Error: S_list and S have different lengths"
    return S_list


def pick_k_set(n, k):
    """
    Randomly pick a k-set S from the set [n].

    :param n: The size of the set [n]
    :param k: The number of elements to pick
    :return: A set of k randomly picked elements from [n]
    """
    # Ensure k is not greater than n
    if k > n:
        raise ValueError("k cannot be greater than n")

    # Generate a list [1, 2, ..., n]
    full_set = list(range(n))

    # Randomly pick k elements from the list
    S = random.sample(full_set, k)

    return S

def construct_W_from_graph(G):
    # Compute the Laplacian matrix of the graph
    L = nx.laplacian_matrix(G).toarray()

    # Compute the identity matrix of the same size as L
    I = np.eye(L.shape[0])

    # Compute W = (I + L)^-1
    W = np.linalg.inv(I + L)

    return W

def construct_tstepFJmodelW_from_graph(t, W):

    # Number of nodes
    n = W.shape[0]

    # Compute a_i for each node
    # a = np.array([1 / (1 + np.sum(W[i, W[i, :] > 0])) for i in range(n)])
    a = np.array([1 / (1 + np.sum(W[i, :])) for i in range(n)])

    # Compute V = diag(1 - a) * W
    V = np.diag(a) @ W

    # Compute V^t * y for Z(t)
    # Vt_y = np.linalg.matrix_power(V, t) @ y

    # Compute sum of V^i * (a odot y) for Z(t) from i = 0 to t-1
    # sum_Vi_ay_Zt = sum([np.linalg.matrix_power(V, i) @ (a * y) for i in range(t)])

    # Compute Z(t)
    # Zt = Vt_y + sum_Vi_ay_Zt

    # Compute sum of V^i * diag(a) for W(t) from i = 0 to t-1
    sum_Vi_da_Wt = sum([np.linalg.matrix_power(V, i) @ np.diag(a) for i in range(t)])

    # Compute W(t)
    Wt = np.linalg.matrix_power(V, t) + sum_Vi_da_Wt

    return Wt





def experiment0(n, W, Y, Z, m, total_iterations, plot=True):
  
    objective_greedy = []
    greedy_selection = greedy_approximation(W, Y, max_iter=total_iterations)
    for t in range(total_iterations):
        objective_value = calculate_objective(Z, W, Y, greedy_selection[:t+1]) / m
        objective_greedy.append(objective_value)
    print("objective_greedy:", objective_greedy)
        
    objective_greedy_appro = []
    greedy_selection_appro = approximate_greedy_approximation(W, Y, max_iter=total_iterations)
    for t in range(total_iterations):
        objective_value = calculate_objective(Z, W, Y, greedy_selection_appro[:t+1]) / m
        objective_greedy_appro.append(objective_value)
    print("objective_greedy_appro:", objective_greedy_appro)

    random_repeat = 10
    objective_random = []
    random_selection = pick_k_set(n, t)
    for t in range(total_iterations):
        random_values = [calculate_objective(Z, W, Y, random_selection[:t+1]) / m for _ in range(random_repeat)]
        mean_objective_value = np.mean(random_values)
        objective_random.append(mean_objective_value)

    print("objective_random:", objective_random)
    print("DONE!")

    return objective_greedy, objective_greedy_appro, objective_random


def experiment1(n, W, Y, Z, m=0, max_iterations=-1, early_stop=1., repeatk=5, plot=False):

    if m==0:
        pass # TODO: compute m from Z

    if max_iterations == -1:
        max_iterations = n

    # def generate_permutations(T, k=5):
    #     permutations = []
    #     for _ in range(k):
    #         # perm = random.shuffle(T)
    #         perm = np.random.permutation(T)
    #         permutations.append(perm)
    #     return permutations

    objective_random = []
    random_seqs = generate_permutations(range(n), k=repeatk)
    t=0
    acc = 0.
    epsilon = 1e-10
    while t < max_iterations and acc <= early_stop - epsilon:
        acc = np.mean([calculate_objective(Z, W, Y, seqs[:t+1]) / m for seqs in random_seqs])
        t += 1
        objective_random.append(acc)
    max_iterations = t
    print("max_iteration:\t",max_iterations)
    print("objective_random", objective_random)

    objective_greedy = []
    greedy_selection = greedy_approximation(W, Y, max_iter=max_iterations)
    for t in range(max_iterations):
        objective_value = calculate_objective(Z, W, Y, greedy_selection[:t+1]) / m
        objective_greedy.append(objective_value)
    print("objective_greedy size:", len(greedy_selection))
    print("objective_greedy:", objective_greedy)
    
    objective_greedy_appro = []
    greedy_selection_appro = approximate_greedy_approximation(W, Y, max_iter=max_iterations)
    print("greedy_selection_appro size:", len(greedy_selection_appro))
    greedy_selection_appro_extns = []
    # t = len(greedy_selection_appro)
    if len(greedy_selection_appro) < max_iterations:
        T = list(set(range(n)) - set(greedy_selection_appro))
        perms = generate_permutations(T, k=repeatk)
        for i in range(repeatk):
            greedy_selection_appro_extns.append(greedy_selection_appro + list(perms[i]) )
    for t in range(max_iterations):
        val = np.mean([calculate_objective(Z, W, Y, S[:t+1]) / m for S in greedy_selection_appro_extns])
        # objective_value = calculate_objective(Z, W, Y, greedy_selection_appro[:t+1]) / m
        objective_greedy_appro.append(val)  
    print("objective_greedy_appro:", objective_greedy_appro)

    return (objective_greedy, objective_greedy_appro, objective_random), max_iterations
        

        
def generate_permutations(T, k=5):
    permutations = []
    for _ in range(k):
        # perm = random.shuffle(T)
        perm = np.random.permutation(T)
        permutations.append(perm)
    return permutations



def experiment5(n, W, Y, Z, m=0, max_iterations=-1, early_stop=1., repeatk=5, plot=False):

    if m==0:
        pass # TODO: compute m from Z

    if max_iterations == -1:
        max_iterations = n

    # def generate_permutations(T, k=5):
    #     permutations = []
    #     for _ in range(k):
    #         # perm = random.shuffle(T)
    #         perm = np.random.permutation(T)
    #         permutations.append(perm) 
    #     return permutations
        

    objective_random = []
    random_seqs = generate_permutations(range(n), k=repeatk)
    t=0
    acc = 0.
    epsilon = 1e-10
    while t < max_iterations and acc <= early_stop - epsilon:
        acc = np.mean([calculate_objective(Z, W, Y, seqs[:t+1]) / m for seqs in random_seqs])
        t += 1
        objective_random.append(acc)
    max_iterations = t
    print("max_iteration:\t",max_iterations)
    print("objective_random:", [ f"{val:.2f}" for val in objective_random])

    objective_greedy_appro = []
    greedy_selection_appro = approximate_greedy_approximation_opt(W, Y, max_iter=max_iterations)
    print("greedy_selection_appro size:", len(greedy_selection_appro))
    greedy_selection_appro_extns = [greedy_selection_appro for _ in range(repeatk)]
    # t = len(greedy_selection_appro)
    if len(greedy_selection_appro) < max_iterations:
        T = list(set(range(n)) - set(greedy_selection_appro))
        perms = generate_permutations(T, k=repeatk)
        for i in range(repeatk):
            greedy_selection_appro_extns[i] = greedy_selection_appro_extns[i]+ list(perms[i])
    for t in range(max_iterations):
        val = np.mean([calculate_objective(Z, W, Y, S[:t+1]) / m for S in greedy_selection_appro_extns])
        # objective_value = calculate_objective(Z, W, Y, greedy_selection_appro[:t+1]) / m
        objective_greedy_appro.append(val)  
    print("objective_greedy_appro:", [ f"{val:.2f}" for val in objective_greedy_appro])

    objective_greedy = []
    greedy_selection = greedy_approximation(W, Y, max_iter=max_iterations)
    for t in range(max_iterations):
        objective_value = calculate_objective(Z, W, Y, greedy_selection[:t+1]) / m
        objective_greedy.append(objective_value)
    print("objective_greedy size:", len(greedy_selection))
    print("objective_greedy:", [ f"{val:.2f}" for val in objective_greedy])

    return (objective_greedy, objective_greedy_appro, objective_random), max_iterations
        


def experiment6(n, W, Y, Z, m=0, max_iterations=-1, early_stop=.99, repeatk=5, plot=False):

    if m==0:
        pass # TODO: compute m from Z

    if max_iterations == -1:
        max_iterations = n



    objective_random = []
    random_seqs = generate_permutations(range(n), k=repeatk)
    t=0
    acc = 0.
    epsilon = 1e-10
    while t < max_iterations and acc <= early_stop - epsilon:
        acc = np.mean([calculate_objective(Z, W, Y, seqs[:t+1]) / m for seqs in random_seqs])
        t += 1
        objective_random.append(acc)
    max_iterations = t
    print("max_iteration:\t",max_iterations)
    print("objective_random", objective_random)

    objective_greedy = []
    greedy_selection = greedy_approximation(W, Y, max_iter=max_iterations)
    for t in range(max_iterations):
        objective_value = calculate_objective(Z, W, Y, greedy_selection[:t+1]) / m
        objective_greedy.append(objective_value)
    print("objective_greedy size:", len(greedy_selection))
    print("objective_greedy:", objective_greedy)
    
    objective_greedy_appro = []
    greedy_selection_appro = approximate_greedy_approximation(W, Y, max_iter=max_iterations)
    print("greedy_selection_appro size:", len(greedy_selection_appro))
    greedy_selection_appro_extns = []
    # t = len(greedy_selection_appro)
    if len(greedy_selection_appro) < max_iterations:
        T = list(set(range(n)) - set(greedy_selection_appro))
        perms = generate_permutations(T, k=repeatk)
        for i in range(repeatk):
            greedy_selection_appro_extns.append(greedy_selection_appro + list(perms[i]) )
    for t in range(max_iterations):
        val = np.mean([calculate_objective(Z, W, Y, S[:t+1]) / m for S in greedy_selection_appro_extns])
        # objective_value = calculate_objective(Z, W, Y, greedy_selection_appro[:t+1]) / m
        objective_greedy_appro.append(val)  
    print("objective_greedy_appro:", objective_greedy_appro)

    return (objective_greedy, objective_greedy_appro, objective_random), max_iterations
        




def experiment7(n, W, Y, Z, m=0, max_iterations=-1, early_stop=1., repeatk=1, plot=False):

    if m==0:
        pass # TODO: compute m from Z

    if max_iterations == -1:
        max_iterations = n

    # def generate_permutations(T, k=5):
    #     permutations = []
    #     for _ in range(k):
    #         # perm = random.shuffle(T)
    #         perm = np.random.permutation(T)
    #         permutations.append(perm) 
    #     return permutations
        

    objective_random = []
    random_seqs = generate_permutations(range(n), k=repeatk)
    t=0
    acc = 0.
    epsilon = 1e-10
    while t < max_iterations and acc <= early_stop - epsilon:
        acc = np.mean([calculate_objective(Z, W, Y, seqs[:t+1]) / m for seqs in random_seqs])
        t += 1
        objective_random.append(acc)
    max_iterations = t
    print("max_iteration:\t",max_iterations)
    print("objective_random:"+ ", ".join([ f"{val:.2f}" for val in objective_random]))

    objective_greedy_appro = []
    greedy_selection_appro = approximate_greedy_approximation_opt(W, Y, max_iter=max_iterations)
    print("greedy_selection_appro size:", len(greedy_selection_appro))
    greedy_selection_appro_extns = [greedy_selection_appro for _ in range(repeatk)]
    # t = len(greedy_selection_appro)
    if len(greedy_selection_appro) < max_iterations:
        T = list(set(range(n)) - set(greedy_selection_appro))
        perms = generate_permutations(T, k=repeatk)
        for i in range(repeatk):
            greedy_selection_appro_extns[i] = greedy_selection_appro_extns[i]+ list(perms[i])
    for t in range(max_iterations):
        val = np.mean([calculate_objective(Z, W, Y, S[:t+1]) / m for S in greedy_selection_appro_extns])
        # objective_value = calculate_objective(Z, W, Y, greedy_selection_appro[:t+1]) / m
        objective_greedy_appro.append(val)  
    print("objective_greedy_appro: " + ", ".join([ f"{val:.2f}" for val in objective_greedy_appro]))
    # print("objective_greedy_appro:", [ f"{val:.2f}" for val in objective_greedy_appro])

    objective_greedy = []
    greedy_selection = greedy_approximation(W, Y, max_iter=max_iterations)
    for t in range(max_iterations):
        objective_value = calculate_objective(Z, W, Y, greedy_selection[:t+1]) / m
        objective_greedy.append(objective_value)
    print("objective_greedy size:", len(greedy_selection))
    print("objective_greedy: " + ", ".join([ f"{val:.2f}" for val in objective_greedy]))

    return (objective_greedy, objective_greedy_appro, objective_random), max_iterations



def get_degree_seq(W, Y, Z):
    W_sum = np.sum(W, axis=1)
    # print(W_sum.shape)  ## shape=(n,)
    W_sum = np.flip(np.argsort(W_sum))
    seq = W_sum

    return seq

def get_error_seq(W, Y, Z):
    Y_sum = np.sum(Y<0, axis=1)
    # print(Y_sum.shape)  ## shape=(n,)
    Y_sum = np.flip(np.argsort(Y_sum))
    seq = Y_sum

    return seq

def get_degreeXerror_seq(W, Y, Z):
    Y_sum = np.sum(Y<0, axis=1)
    W_sum = np.sum(W, axis=1)
    YW_sum = np.multiply(Y_sum, W_sum)
    # print(W_sum.shape)  ## shape=(n,)
    seq = np.flip(np.argsort(YW_sum)) ## descending order

    return seq

AlgorithmList = ["random", "degree", "error", "degreeXerror", "greedy_approx", "greedy"]
AlgorithmCount=len(AlgorithmList)
MASK = [1]*AlgorithmCount


def experiment8(n, W, Y, Z, m=0, max_iterations=-1, early_stop=1., repeatk=1, plot=False, mask=MASK):

    if m==0:
        # compute m from Z
        m = np.sum(Z < 0)

    if max_iterations == -1:
        max_iterations = n



    results = np.zeros([AlgorithmCount, max_iterations])

    ### random selection baseline
    objective_vals = []
    algorithm_id = 0
    if mask[algorithm_id]:
        random_seqs = generate_permutations(range(n), k=repeatk)
        t=0
        acc = 0.
        epsilon = 1e-10
        while t < max_iterations and acc <= early_stop - epsilon:
            acc = np.mean([calculate_objective(Z, W, Y, seqs[:t+1]) / m for seqs in random_seqs])
            t += 1
            objective_vals.append(acc)
        max_iterations = t
        print("objective_random:"+ ", ".join([ f"{val:.2f}" for val in objective_vals]))
        results = results[:, :max_iterations]
        results[algorithm_id] = np.array(objective_vals)
    algorithm_id += 1
    print("max_iteration:\t",max_iterations)

    ### degree based selection baseline
    objective_vals = []
    if mask[algorithm_id]:
        degree_selection = get_degree_seq(W, Y, Z)
        for t in range(max_iterations):
            objective_val = calculate_objective(Z, W, Y, degree_selection[:t+1]) / m
            objective_vals.append(objective_val)
        print("objective_degree:"+ ", ".join([ f"{val:.2f}" for val in objective_vals]))
        results[algorithm_id] = np.array(objective_vals)
    algorithm_id += 1

    ### error rate based selection baseline
    objective_vals = []
    if mask[algorithm_id]:
        selection_seq = get_error_seq(W, Y, Z)
        for t in range(max_iterations):
            objective_val = calculate_objective(Z, W, Y, selection_seq[:t+1]) / m
            objective_vals.append(objective_val)
        print("objective_error:"+ ", ".join([ f"{val:.2f}" for val in objective_vals]))
        results[algorithm_id] = np.array(objective_vals)
    algorithm_id += 1

    ### degreeXerror based selection baseline
    objective_vals = []
    if mask[algorithm_id]:
        selection_seq = get_degreeXerror_seq(W, Y, Z)
        for t in range(max_iterations):
            objective_val = calculate_objective(Z, W, Y, selection_seq[:t+1]) / m
            objective_vals.append(objective_val)
        print("objective_degreeXerror:"+ ", ".join([ f"{val:.2f}" for val in objective_vals]))
        results[algorithm_id] = np.array(objective_vals)
    algorithm_id += 1

    ### greedy approximation selection algorithm
    objective_vals = []
    if mask[algorithm_id]:
        greedy_selection_appro = approximate_greedy_approximation_opt(W, Y, max_iter=max_iterations)
        print("greedy_selection_appro size:", len(greedy_selection_appro))
        greedy_selection_appro_extns = [greedy_selection_appro for _ in range(repeatk)]
        # t = len(greedy_selection_appro)
        if len(greedy_selection_appro) < max_iterations:
            T = list(set(range(n)) - set(greedy_selection_appro))
            perms = generate_permutations(T, k=repeatk)
            for i in range(repeatk):
                greedy_selection_appro_extns[i] = greedy_selection_appro_extns[i]+ list(perms[i])
        for t in range(max_iterations):
            val = np.mean([calculate_objective(Z, W, Y, S[:t+1]) / m for S in greedy_selection_appro_extns])
            # objective_value = calculate_objective(Z, W, Y, greedy_selection_appro[:t+1]) / m
            objective_vals.append(val)  
        print("objective_greedy_appro: " + ", ".join([ f"{val:.2f}" for val in objective_vals]))
        results[algorithm_id] = np.array(objective_vals)
    algorithm_id += 1

    ### greedy selection algorithm
    objective_vals = []
    if mask[algorithm_id]:
        greedy_selection = greedy_approximation(W, Y, max_iter=max_iterations)
        for t in range(max_iterations):
            objective_val = calculate_objective(Z, W, Y, greedy_selection[:t+1]) / m
            objective_vals.append(objective_val)
        print("objective_greedy size:", len(greedy_selection))
        print("objective_greedy: " + ", ".join([ f"{val:.2f}" for val in objective_vals]))
        results[algorithm_id] = np.array(objective_vals)
    algorithm_id += 1

    return results, max_iterations

def plot_results(performance_metrics, dataset_names, title="Comparison of Iterations for Cover Ratio >= 70%", xlabel="Datasets", ylabel="Objective Value", savefig=f"./output/comparison.pdf"):    

    # Data provided by the user
    method_names = ["Greedy", "Approx", "Random"]
    dataset_names = dataset_names
    # metrics = [[1,2,3], [2,3,4], [3,4,5], [4,5,6]] 
    metrics = performance_metrics
    # Setting a lighter color for 'Greedy' method - 'sky blue'
    colors = ['skyblue', 'orange', 'green']

    # Number of datasets
    n_datasets = len(dataset_names)
    # Number of methods
    n_methods = len(method_names)

    # Create a 2D array from the metrics for easy manipulation
    metrics_array = np.array(metrics)

    # Setting the positions and width for the bars
    pos = np.arange(n_datasets)
    bar_width = 0.25

    # Plotting
    plt.figure(figsize=(10, 3))
    # Double the font size for the plot elements
    default_font_size = plt.rcParams['font.size']
    new_font_size = default_font_size * 2 

    # for i in range(n_methods):
    #     plt.bar(pos + i * bar_width, metrics_array[:, i], bar_width, label=method_names[i], color=color)
    for i, color in enumerate(colors):
        bars = plt.bar(pos + i * bar_width, metrics_array[:, i], bar_width, label=method_names[i], color=color)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=13)



    # Adding labels and title
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=18)
    plt.xticks(pos + bar_width, dataset_names, fontsize=20)
    plt.yticks(fontsize=20) 

    # Adding a legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    if savefig is not None:
        plot_fstr = f"./output/comparison.pdf"        
        plt.savefig(plot_fstr, format='pdf')
        print("Save to ==> ", plot_fstr)


    plt.show()


def plot_results8(performance_metrics, dataset_names, method_names=["Greedy", "Approx", "Random"], title="Comparison of Iterations for Cover Ratio >= 70%", xlabel="Datasets", ylabel="Objective Value", savefig=f"./output/comparison.pdf"):    

    # Data provided by the user
    method_names = method_names
    dataset_names = dataset_names
    # metrics = [[1,2,3], [2,3,4], [3,4,5], [4,5,6]] 
    metrics = performance_metrics
    # Setting a lighter color for 'Greedy' method - 'sky blue'
    colors = ['skyblue', 'orange', 'green']

    # Number of datasets
    n_datasets = len(dataset_names)
    # Number of methods
    n_methods = len(method_names)

    # Create a 2D array from the metrics for easy manipulation
    metrics_array = np.array(metrics)

    # Setting the positions and width for the bars
    pos = np.arange(n_datasets)
    bar_width = 0.25

    # Plotting
    plt.figure(figsize=(10, 3))
    # Double the font size for the plot elements
    default_font_size = plt.rcParams['font.size']
    new_font_size = default_font_size * 2 

    # for i in range(n_methods):
    #     plt.bar(pos + i * bar_width, metrics_array[:, i], bar_width, label=method_names[i], color=color)
    for i, color in enumerate(colors):
        bars = plt.bar(pos + i * bar_width, metrics_array[:, i], bar_width, label=method_names[i], color=color)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=13)



    # Adding labels and title
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=18)
    plt.xticks(pos + bar_width, dataset_names, fontsize=20)
    plt.yticks(fontsize=20) 

    # Adding a legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    if savefig is not None:
        plot_fstr = f"./output/comparison.pdf"        
        plt.savefig(plot_fstr, format='pdf')
        print("Save to ==> ", plot_fstr)


    plt.show()