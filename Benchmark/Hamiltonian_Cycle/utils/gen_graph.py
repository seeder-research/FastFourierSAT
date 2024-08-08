import networkx as nx
import numpy as np

def generate_bin_graph(num_vertex, rho, seed = 0):
    G_route = nx.gnp_random_graph(num_vertex, rho, seed)
    G_cons = nx.complement(G_route)
    V = np.arange(1, num_vertex**2+1).reshape(num_vertex,num_vertex)
    CNF = []
    for e in G_cons.edges:
        v0 = V[:,e[0]]
        v1 = V[:,e[1]]
        for i in range(len(v0)-1):
            CNF.append([-v0[i], -v1[i+1]])
            CNF.append([-v1[i], -v0[i+1]])
        CNF.append([-v0[-1], -v1[0]])
        CNF.append([-v1[-1], -v0[0]])
    EO = []
    for i in range(num_vertex):
        EO.append(V[i,:].tolist())
        EO.append(V[:,i].tolist())
    return CNF, EO

def generate_reg_graph(num_vertex, degree, seed = 0):
    G_route = nx.random_regular_graph(degree, num_vertex, seed)
    G_cons = nx.complement(G_route)
    V = np.arange(1, num_vertex**2+1).reshape(num_vertex,num_vertex)
    CNF = []
    for e in G_cons.edges:
        v0 = V[:,e[0]]
        v1 = V[:,e[1]]
        for i in range(len(v0)-1):
            CNF.append([-v0[i], -v1[i+1]])
            CNF.append([-v1[i], -v0[i+1]])
        CNF.append([-v0[-1], -v1[0]])
        CNF.append([-v1[-1], -v0[0]])
    EO = []
    for i in range(num_vertex):
        EO.append(V[i,:].tolist())
        EO.append(V[:,i].tolist())
    return CNF, EO