from pred_v2 import *
import sympy as sp
import networkx as nx
import numpy as np  


class flh_3:
    def __init__(self, alpha, m):
        # Constantes
        self.alpha = alpha # Alpha que se va a usar, en principio siempre mayor que 0.5
        # self.num_threads = num_threads
        self.m = m 
        # Variables de cada hilo
        self.predictions = [[]] # Array de las predicciones, para consultar y futuras actualizaciones, además no se desperdicia memoria
        self.thread_times = [1] # Próximo tiempo para cada hilo (con el que se ejecuta el algoritmo)
        self.probs = [1.0] # Probabilidades actuales de cada hilo
        self.lifetimes = [5] # 5 es el primer lifetime (2^(0 + 2) + 1)
        self.thread_index = [0] # Indice de cada hilo de ejecucion
        # Variables de todos los hilos
        self.last_flows = []
        self.total_time = 1 # Tiempo actual, de la última ejecución
        self.last_prediction = [[0.5]]
    # flow es el flujo que se ha revelado de la última situación
    def execute(self, flow):
        pred = [0.0]
        if(self.total_time != 1):
            self.last_flows.append(flow)
            # Se actualizan las probabilidades y se actualizan los parámetros
            new_rel_prob_aux = 1/(self.total_time - 1) # La probabilidad relativa del nuevo hilo
            rel_probs_aux = [] # Las probabilidades relativas de los hilos que ya han actuado
            rel_probs_aux_alive = []
            seen = 0
            deaths = []
            for prob in self.probs:
                index_prob = (self.probs[seen:].index(prob)) + seen
                aux = prob * math.exp(-1 * self.alpha * resource_f_simple(self.predictions[index_prob] , self.m, self.last_flows[len(self.last_flows) - 2]))
                if(math.isnan(aux)):
                    aux = 1.0
                rel_probs_aux.append(aux)
                self.lifetimes[index_prob] -= 1  
                # Se eliminan los hilos que han muerto (PRUNE)
                if(self.lifetimes[index_prob] <= 0):
                    deaths.append(index_prob)
                    continue
                rel_probs_aux_alive.append(aux)
                seen += 1
                # PRUNE
            for i in range(len(deaths) - 1, -1, -1):
                    self.thread_times.pop(deaths[i])
                    self.probs.pop(deaths[i])
                    self.lifetimes.pop(deaths[i])
                    self.thread_index.pop(deaths[i])
                    self.predictions.pop(deaths[i])
                    continue
            if(sum(rel_probs_aux) == 0):
                rel_probs_aux_alive = [0.0] * len(rel_probs_aux_alive)
            else:    
                rel_probs_aux_alive = [rel_prob/sum(rel_probs_aux) for rel_prob in rel_probs_aux_alive]                
            rel_probs_aux_alive.append(new_rel_prob_aux) # Se añade la del nuevo hilo
            self.probs = [rel_prob/sum(rel_probs_aux_alive) for rel_prob in rel_probs_aux_alive] # Se actualizan las nuevas probabilidades para la próxima ejecución
            # Parámetros del nuevo hilos
            self.thread_times.append(1)
            self.thread_index.append(len(self.last_flows) - 1)
            i = self.total_time
            k = 0
            while((i % 2) == 0):
                k += 1
                i /= 2
            new_lifetime = (2 ** (k + 2)) + 1
            self.lifetimes.append(new_lifetime)
            # Se actualizan los flujos (para no almacenarlos todos)
            self.last_flows = self.last_flows[min(self.thread_index) : len(self.last_flows)]
            # Se actualizan todos los índices teniendo en cuentas los hilos que se hayan eliminado
            self.thread_index = [index - min(self.thread_index) for index in self.thread_index]
            # Se ejecuta el algoritmo
            self.predictions.append([0.0]) # Se reinician las predicciones
            for i in range(0, len(self.lifetimes)):
                if(i < len(self.lifetimes)):
                    aux = grad_descent_antiguo(self.predictions[i], self.thread_times[i], flow, self.m) 
                else:
                    aux = grad_descent_antiguo(self.last_prediction, self.thread_times[i], flow, self.m)
                pred = [(self.probs[i] * aux[0]) + pred[0]] 
                self.thread_times[i] += 1 # Para las que ya han actuado   
                self.predictions[i] = aux
        else:
            self.last_flows.append(flow) # Cuidado cuando se tenga que reiniciar
            pred = grad_descent_antiguo([0.5], self.thread_times[0], flow, self.m)
            self.thread_times[self.total_time - 1] += 1
            self.lifetimes[self.total_time - 1] -=1
            self.predictions[0] = pred     
        self.total_time += 1 # Se actualiza el tiempo
        self.last_prediction = pred
        return pred


    def get_graph(self):
        return self.G

    def get_alpha(self):
        return self.alpha

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
    
    def get_total_thread_lifetime(self):
        return self.lifetimes