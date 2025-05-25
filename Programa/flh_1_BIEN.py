from pred_v2 import *
import sympy as sp
import networkx as nx
import numpy as np  

class flh:
    # Definimos el objeto
    def __init__(self, alpha ,num_threads, m):
        # Constantes
        self.alpha = alpha 
        self.num_threads = num_threads
        self.m = m
        # Variables de cada hilo
        self.predictions = [[]]*num_threads # Array de las predicciones, para consultar y futuras actualizaciones, además no se desperdicia memoria
        self.thread_times = [2] + [[]]*(num_threads - 1) # Próximo tiempo para cada hilo (con el que se ejecuta el algoritmo)
        self.probs = [1.0] + [[]]*(num_threads - 1) # Probabilidades actuales de cada hilo
        self.thread_index = [0] + [[]]*(num_threads - 1) # Indice de cada hilo de ejecucion
        # Variables de todos los hilos
        self.last_flows = [[]]*num_threads
        self.total_time = 1 # Tiempo actual, de la última ejecución
    # flow es el flujo que se ha revelado de la última situación
    def execute(self, flow):
        pred = [0.0]
        self.last_flows[(self.total_time - 1)] = flow # Cuidado cuando se tenga que reiniciar
        if(self.total_time != 1):
            # Se actualizan las probabilidades y se actualizan los parámetros
            new_rel_prob_aux = 1/(self.total_time) # La probabilidad relativa del nuevo hilo
            rel_probs_aux = [0]*(self.total_time - 1) # Las probabilidades relativas de los hilos que ya han actuado
            for i in range(0, len(rel_probs_aux)):
                rel_probs_aux[i] = self.probs[i] * math.exp(-1 * self.alpha * resource_f_simple(self.predictions[i], self.m, self.last_flows[self.total_time - 2]))
                self.thread_times[i] += 1 # Para las que ya han actuado
            if(sum(rel_probs_aux) == 0):
                rel_probs_aux = [1/len(rel_probs_aux)] * len(rel_probs_aux)
            else:
                rel_probs_aux = [rel_prob/sum(rel_probs_aux) for rel_prob in rel_probs_aux]
            self.probs = [rel_prob * (1 - (1/self.total_time)) for rel_prob in rel_probs_aux] # Se actualizan las nuevas probabilidades para la próxima ejecución
            self.probs.append(new_rel_prob_aux)
            # Parámetros del nuevo hilos
            self.thread_times[self.total_time - 1] = 2
            self.thread_index[self.total_time - 1] = self.total_time - 1
            # Se ejecuta el algoritmo
            for i in range(0, (self.total_time)):
                if(i != (self.total_time - 1)):
                    aux = grad_descent_antiguo(self.predictions[i], self.thread_times[i], self.last_flows[self.total_time - 2], self.m)
                else:
                    aux = grad_descent_antiguo([0.5], self.thread_times[i], self.last_flows[self.total_time - 2], self.m)
                pred = [(self.probs[i] * aux[0]) + pred[0]] 
                self.predictions[i] = aux
        else:
            self.last_flows[(self.total_time - 1)] = flow 
            self.thread_times[self.total_time - 1] += 1
            pred =  [0.5]
            self.predictions[0] = pred
        self.total_time += 1 # Se actualiza el tiempo
        return pred


    def get_graph(self):
        return self.G

    def get_alpha(self):
        return self.alpha

    def get_num_threads(self):
        return self.num_threads

    def get_predictions(self):
        return self.predictions

    def get_thread_times(self):
        return self.thread_times

    def get_probs(self):
        return self.probs

    def get_thread_index(self):
        return self.thread_index

    def get_last_flows(self):
        return self.last_flows

    def get_total_time(self):
        return self.total_time