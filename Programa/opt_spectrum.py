import numpy as np  
from scipy.optimize import minimize
import sympy as sp
import networkx as nx
import random
import math

# Función para crear un subgrafo, con distancias (en tiempo) menores a max_time desde el nodo origen hasta cualquier nodo del grafo. Devuelve los nodos del subgrafo.
def getSubgraph_time(G ,node_ini ,max_time ,actual_time, visitados):
    vecinos = G.neighbors(node_ini)
    vecinos_cerca = []
    tiempos_vecinos = []
    visitados.add(node_ini)
    for neighbor in vecinos:
        t_aux = (G[node_ini][neighbor]["distance"] / (G[node_ini][neighbor]["vel"] * 1000)) * 60
        if(t_aux < (max_time - actual_time)):
            vecinos_cerca.append(neighbor)
            tiempos_vecinos.append(t_aux + actual_time)
    for i in range(0, len(vecinos_cerca)):
        if((vecinos_cerca[i] in visitados) == False):
            vecinos_cerca += getSubgraph_time(G ,vecinos_cerca[i] ,max_time=max_time , actual_time=tiempos_vecinos[i], visitados=set().union(visitados))
    return list(set(vecinos_cerca))



############################################## Funciones de pruebas pasadas #######################################
def G_num(p , vars, G):
    G_num = G.copy()
    sustituciones = dict(zip(vars, p))
    for edge in G_num.edges(data=True):
        edge[2]["weight"] = float(edge[2]["weight"].subs(sustituciones))
    return G_num

def max_degree(G_num):
    maxdegree = 0
    for node in G_num.nodes:
        degree = 0
        for edge in G_num.edges(node, data=True):
            degree += edge[2]["weight"]
        if(degree > maxdegree):
            maxdegree = degree
    return maxdegree
