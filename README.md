# DISEÑO E IMPLEMENTACIÓN DE UN ALGORITMO DE SLICING FIABLE PARA VEHÍCULOS CONECTADOS EN REDES B5G

> Escuela Técnica Superior de Ingenieros de Telecomunicación – UPM  
> Curso 2024/2025
## 👤 Autor

- Adrián Díaz Gómez

## 📝 Descripción

Este repositorio contiene la implementación de los algoritmos desarrollados en el Trabajo de Fin de Grado titulado **“Diseño e implementación de un algoritmo de slicing fiable para vehículos conectados en redes B5G”**.  
El código permite simular diferentes algoritmos de predicción de tráfico y comparar sus resultados mediante visualización gráfica.

## 🧰 Tecnologías y herramientas

- **Lenguaje principal:** Python 3
- **Librerías principales utilizadas:** `networkx`, `numpy`, `scipy`, `matplotlib`, `json`
- **Entorno de desarrollo:** VS Code

## 🚀 Instrucciones de uso
A continuación se detallan los pasos para comprobar los resultados:

- El primer paso es clonar el repositorio: git clone (https://github.com/G0ATEN/TFG.git)
- Seguidamente, se podrá acceder a las funciones y a los algoritmos desarrollados.
- En los archivos `flh_1BIEN.py` y `flh_3BIEN.py` se encuentran implementados los algoritmos FLH y FLH2 respectivamente. Estos algoritmos se implementan a partir de una clase, y el método execute y devuelve el resultado de la ejecución del algoritmo.
- En el archivo `pred_v2.py` se encuentran las funciones grad_descent y grad_descent_antiguo, que representan los algoritmos de OGD con salto fijo y salto variable respectivamente.
- El archivo `torino_simulation.py` permite simular los distintos algoritmos y observar los resultados.

