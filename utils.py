import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
plt.rcParams['savefig.format'] = 'pdf' 

from scipy.io import mmread
import pickle
import networkx as nx



def initialize_W(n, rho, normalize=True):
    """
    Initialize a random nonnegative n x n matrix with sparsity controlled by rho.

    :param n: Size of the matrix (n x n)
    :param rho: Sparsity factor (between 0 and 1), with rho=1 meaning a fully dense matrix
    :return: Randomly generated nonnegative matrix W
    """
    # Create a random matrix with values between 0 and 1
    W = np.random.rand(n, n)

    # Make the matrix sparse by setting a fraction of elements to zero
    if rho < 1:
        mask = np.random.rand(n, n) > rho
        W[mask] = 0
        
    if normalize:
        row_sums = W.sum(axis=1, keepdims=True)
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
    # Create a binary matrix based on Bernoulli distribution
    y_binary = np.random.binomial(1, p, (n, k))

    y = 2 * y_binary - 1

    return y


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
            condition1 = np.dot(W[i, :], Y[:, a]) < 0

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

    # E = W @ (1-2*Y)
    for i in range(n):
        sum1 = np.sum([ W[i,j] for j in S])
        sum2 = np.sum( [ W[i,j] * (1. - 2.*Y[j,0] )- W[i,t]  for j in range(n) if j not in S and j != t ] ) 
        phi = sum1 + sum2
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

    return total_sum

def approximate_greedy_approximation(W, Y, max_iter=10):
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
            continue

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

def construct_finiteFJmodelW_from_graph(t, W, y):

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


def experiment1(n, W, Y, Z, m=0, max_iterations=-1, early_stop=1., plot=True):

    if m==0:
        pass # TODO: compute m from Z

    if max_iterations == -1:
        max_iterations = n

    def generate_permutations(n, k=5):
        permutations = []
        for _ in range(k):
            perm = np.random.permutation(n)
            permutations.append(perm)
        return permutations

    objective_random = []
    random_seqs = generate_permutations(n)
    t=0
    acc = 0.
    epsilon = 1e-10
    while t < max_iterations and acc < early_stop - epsilon:
        acc = np.mean([calculate_objective(Z, W, Y, seqs[:t+1]) / m for seqs in random_seqs])
        t += 1
        objective_random.append(acc)
    max_iterations = t
    print("max_iteration:\t",max_iterations)

    objective_greedy = []
    greedy_selection = greedy_approximation(W, Y, max_iter=max_iterations)
    for t in range(max_iterations):
        objective_value = calculate_objective(Z, W, Y, greedy_selection[:t+1]) / m
        objective_greedy.append(objective_value)
    print("objective_greedy:", objective_greedy)
    
    objective_greedy_appro = []
    greedy_selection_appro = approximate_greedy_approximation(W, Y, max_iter=max_iterations)
    for t in range(max_iterations):
        objective_value = calculate_objective(Z, W, Y, greedy_selection_appro[:t+1]) / m
        objective_greedy_appro.append(objective_value)
    print("objective_greedy_appro:", objective_greedy_appro)

    return (objective_greedy, objective_greedy_appro, objective_random), max_iterations
        




