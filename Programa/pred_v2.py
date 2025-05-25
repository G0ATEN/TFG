import numpy as np  
from scipy.optimize import minimize
import sympy as sp
import networkx as nx
import csv
from opt_spectrum import *
from scipy.integrate import quad
import math as math



# Función de pérdida
def resource_f_simple(params, m ,flow):
    return float((((m) * params[0]) - flow)**2)

# Resultado óptimo (a posteriori)
def opt_result_simple(m, flow):
    x = flow / m
    if(x > 1.0):
        return [1.0]
    return [x]

# OGD de salto fijo
def grad_descent(last_pred, last_flow, m, step):
    grad = 2 * m * (((m) * last_pred[0]) - last_flow)
    new_pred = last_pred[0] - step * grad
    new_pred = np.clip(new_pred, 0.0, 1.0)
    return [float(new_pred)]

# OGD con actualización de salto
def grad_descent_antiguo(last_pred, t, last_flow, m):
    D = 1
    G = 2 * (m**2)
    step = D/(G * math.sqrt(t))
    grad = 2 * m * (((m) * last_pred[0]) - last_flow)
    new_pred = last_pred[0] - step * grad
    new_pred = np.clip(new_pred, 0.0, 1.0)
    return [float(new_pred)]

# Estas funciones son para el cálculo del Regret Estático.
# Función global, que suma todas las funciones de pérdida hasta un horizonte T. Con una misma x.
def global_resource_f_simple(x, m, flows_T):
    T = len(flows_T)
    x = x[0] # Para el minimize
    square_flows = [flow**2 for flow in flows_T]
    res = ((T * (m * x)** 2) + (sum(square_flows)) - (2 * m * x * sum(flows_T)))
    return res

# Devuelve el resultado que minimiza la función global
def global_opt_result_simple(m, flows_T):
    res = list(minimize(global_resource_f_simple, [1.0], args=(m, flows_T), bounds=[(0,1)])["x"])
    res = float(res[0])
    return [res]

##################################################### Funciones de pruebas pasadas. Para cáculos con múltiples variables y otros algoritos OCO ######################

def getFlows(csv_route, precision):
    flows = []
    with open(csv_route, mode='r') as archivo:
        data = csv.DictReader(archivo)  
        for i, fila in enumerate(data):
            if i % precision == 0: 
                flows.append(float(fila['flow'])) 
    return flows


def resource_f(params, vars ,G , flow):
    G_n = G_num(params, vars, G)
    lambda2 = nx.laplacian_spectrum(G_n, weight="weight")[1]
    return float((lambda2/2 - flow)**2)


def opt_result(vars, G, flow):
    start_params = [0.5]*len(vars)
    res = list(minimize(resource_f, start_params, args = (vars,G,flow), bounds=[(0,1)]*len(vars))["x"])
    res = [float(e) for e in res]
    return res


def global_resource_f_T(params, vars, G, flow,T):
    G_n = G_num(params, vars, G)
    lambda2 = nx.laplacian_spectrum(G_n, weight="weight")[1]
    return [((lambda2/2)**2 + flow**2 -(lambda2*flow))*T]


def global_opt_result(vars, G, flow, T):
    start_params = [0.5]*len(vars)
    res = list(minimize(global_resource_f_T, start_params, args = (vars,G,flow,T), bounds=[(0,1)]*len(vars))["x"])
    res = [float(e) for e in res]
    return res


def ewoo_pred(alpha, flows_t_1, t,vars,G, last_result):
    alpha = alpha # Se puede cambiar a > 0.5
    G_n = G_num([1.0], vars, G)
    lambda2_m = nx.laplacian_spectrum(G_n,weight="weight")[1]
    k1 = ((t - 1)/4)*(lambda2_m**2)
    k2 = lambda2_m*sum(flows_t_1)
    factor = 1
    num, error_num = quad(integrand_num, 0, 1, args=(alpha, k1, k2, factor))
    den, error_den = quad(integrand_den, 0, 1, args=(alpha, k1, k2, factor))
    return [num / den]

def ewoo_pred_simple(alpha, flows_t_1, t, lambda2_m, last_result):
    alpha = alpha # Se puede cambiar a > 0.5
    k1 = ((t - 1)/4)*(lambda2_m**2)
    k2 = lambda2_m*sum(flows_t_1)
    factor = 1
    num, error_num = quad(integrand_num, 0, 1, args=(alpha, k1, k2, factor))
    den, error_den = quad(integrand_den, 0, 1, args=(alpha, k1, k2, factor))

    factor_den = 1.0
    while(math.isnan(den) or den == 0):
        if(den == 0):
            factor_den *= 100
        else:    
            factor_den /200
        den, error_den = quad(integrand_den, 0, 1, args=(alpha, k1, k2, factor_den))
        if(math.isinf(factor_den)):
            return last_result   
    factor_num = 1.0
    while(math.isnan(num) or num == 0):
        if(num == 0):
            factor_num *= 100
        else:    
            factor_num /200
        num, error_num = quad(integrand_num, 0, 1, args=(alpha, k1, k2, factor_num))
        if(math.isinf(factor_num)):
            return last_result
        print("NUM",num)
    print("NUM",num)
    print("DEN",den)
    if(factor_num*num/(den*factor_den) > 1.0):
        return  last_result
    return [factor_num*num/(den*factor_den)]


def multiple_grad_descent(last_pred, t, last_flow, m):
    new_p = last_pred
    for i in range(0, 50):
        new_p = grad_descent_antiguo(new_p, t, last_flow, m)
    return new_p    


def extra_grad_descent(last_pred, t, last_flow, m):
    beta = (2 * m **2)
    step = 1/(beta * 4)
    grad = 2 * m * (((m) * last_pred[0]) - last_flow)
    new_pred = last_pred[0] - step * grad
    new_pred = np.clip(new_pred, 0.0, 1.0)
    new_grad = 2 * m * (((m) * new_pred) - last_flow)
    new_new_pred = new_pred - step * grad
    new_new_pred = np.clip(new_new_pred, 0.0, 1.0)
    return [float(new_new_pred)]

def integrand_func(x, a, k1, k2):
    return (x * np.exp(( -1 * a * k1 * ((x - (k2/(2*k1))) ** 2))))

def integrand_num(x, a, k1, k2, factor_norm):
    return (x * np.exp(( -1 * a * k1 * ((x - (k2/(2*k1))) ** 2)))) / factor_norm 


def integrand_den(x, a, k1, k2, factor_norm):
    return (np.exp(( -1 * a * k1 * ((x - (k2/(2*k1))) ** 2)))) / factor_norm 
