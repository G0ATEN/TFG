import numpy as np  
from scipy.optimize import minimize
import sympy as sp
import networkx as nx
import csv
from opt_spectrum import *
from pred_v2 import *
from sympy.plotting import plot3d
from sympy.plotting import plot
import random
import time
import matplotlib.pyplot as plt
import os
from flh_1_BIEN import *
from flh_3_BIEN import * 
import csv
from collections import defaultdict
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

# # Especificar la ciudad
import json
import networkx as nx
import matplotlib.pyplot as plt


x = sp.symbols('x')
vars = [x]

# 1. Cargar el archivo JSON
with open('TORINO.json', 'r') as f:
    data = json.load(f)

# 2. Crear un grafo simple no dirigido
G = nx.DiGraph()

# 3. Agregar nodos con coordenadas como identificadores únicos
for node in data["nodes"]:
    coord = tuple(node["id"])
    G.add_node(coord)


# 4. Agregar aristas usando sólo las coordenadas. Se crean aristas en ambos sentidos para calcular el corte mínimo correctamente (si no networkx puede dar errores).
for link in data["links"]:
    source = tuple(link["source"])
    target = tuple(link["target"])
    distance = link.get("distance", 0)  
    max_toded = link.get("max_toded", 0) # Capacidad instantánea
    vel = link.get("max_speed",0)
    t = (distance * 0.001 / vel) * 60
    weight = max_toded / t # Capacidad de las aristas
    G.add_edge(source, target, distance=distance, max_toded=max_toded, vel=vel, T = t, weight=weight)
    G.add_edge(target, source, distance=distance, max_toded=max_toded, vel=vel, T = t, weight=weight)

# Creación del subgrafo que se menciona en el TFG.
start_node = (45.0622348,7.668573)
dest_node = (45.0705814, 7.6817539)
G_sub = G.subgraph(getSubgraph_time(G,start_node, 2.5, 0, set())).copy()
G = G_sub

######################################### Dibujar el grafo (descomentar) ##################################################

# # 5. Extraer las posiciones de los nodos para dibujar
# pos = {node: (node[1], node[0]) for node in G.nodes()}  # (x, y)

# # 9. Dibujar el grafo con colores según capacidad
# fig, ax = plt.subplots(figsize=(12, 12))

# capacities = [G[u][v]["weight"] for u, v in G.edges()]
# # print("CAPS", capacities)
# edges = nx.draw_networkx_edges(
#     G, pos,
#     edge_color=capacities,
#     edge_cmap=plt.cm.viridis,
#     edge_vmin=min(capacities),
#     edge_vmax=max(capacities),
#     ax=ax
# )

# node_colors = ['lightgreen' if n == start_node  else 'lightblue' if n == dest_node else 'red' for n in G.nodes]


# node_sizes = [50 if n == start_node or n == dest_node else 1 for n in G.nodes]
# nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,ax=ax)

# sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(capacities), vmax=max(capacities)))
# plt.colorbar(sm, ax=ax, label='Capacidad Instantánea Máxima')

# plt.title("Subgrafo de Torino")

# # 10. Tooltip al pasar el ratón por nodos
# node_pos = {node: pos[node] for node in G.nodes()}
# tooltip = ax.text(0, 0, "", va="bottom", ha="right", color="black", fontsize=10,
#                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='grey'))

# def on_hover(event):
#     if event.inaxes == ax:
#         for node, (x, y) in node_pos.items():
#             if abs(event.xdata - x) < 0.000005 and abs(event.ydata - y) < 0.000005:
#                 tooltip.set_text(f"ID: {node}")
#                 tooltip.set_position((x, y))
#                 tooltip.set_visible(True)
#                 fig.canvas.draw_idle()
#                 return
#         tooltip.set_visible(False)
#         fig.canvas.draw_idle()

# fig.canvas.mpl_connect("motion_notify_event", on_hover)

# plt.show()

############################################################################################################

# Obtención de flujos a partir de los datos (ya procesados)
archivo_csv = 'flujo_medio_por_tramo.csv'

datos_procesados = []
with open(archivo_csv, newline='', encoding='utf-8') as csvfile:
    lector = csv.reader(csvfile)
    next(lector)  # Saltar la cabecera
    for fila in lector:
        valor = float(fila[-1]) * 20 # Cambiar entre 20 y 10 para cambiar el escalado
        datos_procesados.append(valor)
flows = datos_procesados



# Flujo medio (descomentar)
# mean_flow = sum(flows)/len(flows)
# print("FLUJO MEDIO", mean_flow)



cut_value = nx.maximum_flow_value(G, start_node, dest_node, capacity="weight")
print("CUT", cut_value)

# Cálculo de los valores óptimos reales para los flujos dados
opt_values = [[]]*(len(flows))
for i in range(len(flows)): 
    opt_values[i] = opt_result_simple(m = cut_value, flow=flows[i])

# Corte mínimo global (descomentar)
# G = nx.to_undirected(G)
# cut_value = nx.stoer_wagner(G,weight="weight")[0]

# Valores óptimos para el algoritmo. Por si se usa una capacidad menor a la real.
opt_values_2 = [[]]*(len(flows))
for i in range(len(flows)): 
    opt_values_2[i] = opt_result_simple(m = cut_value, flow=flows[i])


#  Ajustar la cantidad de datos de flujo que se van a usar.
flows = flows[0:len(flows) - 1]

#  DIbujar los flujos
# x = [5 * i for i in range(0, len(flows))]
# plt.plot(x, flows, label="Flujos Normalizados", color="purple")
# plt.xlabel("t (tiempo en minutos)", fontsize=14, fontweight='bold')
# plt.ylabel("Flujos escalados (x10)",fontsize=14, fontweight='bold')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# mean_flow = sum(flows)/len(flows)
# print("FLUJO MEDIO", mean_flow)


# Cálculo de los valores de predicción. 
pred_values = [[]]*len(flows)
pred_values[0] = [0.5]
t_time = 1 
# Alpha de alpha-exp concave (cambiar a 100 para ver el resultado experimental)
alpha = (6.74 * 0.000001)
# alpha = 100
flh_1 = flh(alpha, len(flows), cut_value)
flh_2 = flh_3(alpha, cut_value)

for i in range(1, len(flows)):
    # Comentar y descomentar las siguientes líneas en función del algorimo a ejecutar.
    
    # OGD de salto fijo
    # pred_values[i] = grad_descent(pred_values[i-1], flows[i-1], m=cut_value, step=1/(2 * (2 * cut_value**2)))
    
    # OGD con salto variable
    # pred_values[i] = grad_descent_antiguo(pred_values[i-1], t_time, flows[i - 1], cut_value)
    # t_time+=1 # Actualización de t

    # FLH
    # pred_values[i] = flh_1.execute(flows[i - 1])
    
    # FLH2
    pred_values[i] = flh_2.execute(flows[i - 1])


    # print("Iteración", i)

# Cálculo de las pérdidas del algoritmo
f_results = [[]]*len(flows)
for i in range(0, len(flows)):
    f_results[i] = resource_f_simple(pred_values[i], cut_value, flows[i])

# Dibujar gráficas de resultados
x = [5 * i for i in range(0, len(flows))]
# plt.plot(x, opt_values[0:len(pred_values)], label="Valores óptimos", color="black")
plt.plot(x, opt_values_2[0:len(pred_values)], label="Valores óptimos", color="blue")
plt.plot(x, pred_values, label="Predicción", color="red")
plt.xlabel("t (tiempo en minutos)", fontsize=14, fontweight='bold')
plt.ylabel("x_t", fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()  


# Cálculo del error
error = []
for i in range(0, len(pred_values)):
    error.append(abs(pred_values[i][0] - opt_values[i][0]))
t = [1 * i for i in range(0, len(flows))]     
plt.plot(t, error, label="Recursos", color="green")
plt.xlabel("t (iteraciones)",  fontsize=14, fontweight='bold')
plt.ylabel("Error en cada iteración |xt - xt*|", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show() 

mean_error = sum(error)/len(error)
print("ERROR MEDIO", mean_error)



############################## Cálculo y represenación del Regret Dinámico (si aplica) ####################################
# d_regret = []
# for i in range(0, len(pred_values)):
#     d_regret.append(sum(f_results[0: i + 1]))
#     # print("Iteración: ", i)


# L = 2 * (cut_value ** 2)
# mu = L

# lambda_1 = 1
# lambda_1p = 1

# sum = 0
# c_2 = [[]]*len(flows)


# sum = 0
# cotas = [[]]* len(flows)
# for i in range(0, len(flows)):
#     for i in range(1, i + 1):
#         sum += ((opt_values_2[i][0] - opt_values_2[i-1][0]) ** 2)
#     c_2[i] = sum
#     cotas[i] = (L * (3 * sum + 0.75 * ((pred_values[0][0] - opt_values_2[0][0])**2)))


# plt.plot(t, d_regret, label="Regret Dinámico", color="red")
# plt.xlabel("t (iteraciones)",  fontsize=14, fontweight='bold')
# plt.ylabel("D_Regret(t)",  fontsize=14, fontweight='bold')
# plt.legend(fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid(True)
# plt.tight_layout()
# plt.show()  

# plt.plot(t, d_regret, label="Regret Dinámico", color="red")
# plt.plot(t, cotas, label = "Cota del Regret dinámico", color = "blue")
# plt.xlabel("t (iteraciones)",  fontsize=14, fontweight='bold')
# plt.legend(fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid(True)
# plt.tight_layout()
# plt.show()  

################################ Cálculo y represenación del Regret Estático (si aplica) ####################################
regret_T = [0] * len(flows)

for i in range(0, len(regret_T)):
    f_losses_t = sum(f_results[0: i + 1])
    global_opt_t = global_opt_result_simple(cut_value, flows[0: i + 1])
    global_losses_t = global_resource_f_simple(global_opt_t, cut_value, flows[0: i + 1]) 
    regret_T[i] = (f_losses_t - global_losses_t) 
    # print("Iteración: ", i)

G_c = (2 * cut_value**2)
D = 1
cotas = [1.5 * D * G_c * math.sqrt(t) for t in range(1, len(pred_values) + 1)]

plt.plot(t, regret_T, label = "Regret Estático", color= "red")
plt.xlabel("t (iteraciones)",  fontsize=14, fontweight='bold')
plt.ylabel("R_Estático(t)",  fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


plt.plot(t, regret_T, label = "Regret Estático", color= "red")
plt.plot(t, cotas, label="Cota del Regret Estático", color= "blue")
plt.xlabel("t (iteraciones)",  fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

