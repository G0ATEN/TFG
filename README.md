# Diseño e Implementación de un Algoritmo de Slicing Fiable para Vehículos Conectados en Redes B5G

**Trabajo de Fin de Grado – Curso 2024/2025**  
**Escuela Técnica Superior de Ingenieros de Telecomunicación – Universidad Politécnica de Madrid**  

## 👤 Autor
**Adrián Díaz Gómez**

## 📝 Descripción

Este repositorio contiene la implementación de los algoritmos desarrollados en el Trabajo de Fin de Grado titulado **“Diseño e implementación de un algoritmo de slicing fiable para vehículos conectados en redes B5G”**.  
El código permite simular distintos algoritmos de predicción de tráfico y comparar sus resultados mediante visualización gráfica.

## 🧰 Tecnologías utilizadas

- **Lenguaje principal**: Python 3
- **Librerías**:
  - `networkx`
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `json`
- **Entorno de desarrollo**: Visual Studio Code

## 🚀 Instrucciones de uso
   git clone https://github.com/G0ATEN/TFG.git

## 🗂️ Descripción de los archivos
- En los archivos `flh_1BIEN.py` y `flh_3BIEN.py` se encuentran implementados los algoritmos FLH y FLH2 respectivamente. Estos algoritmos se implementan a partir de una clase, y el método execute y devuelve el resultado de la ejecución del algoritmo.
- En el archivo `pred_v2.py` se encuentran las funciones grad_descent y grad_descent_antiguo, que representan los algoritmos de OGD con tasa fija y tasa variable respectivamente.
- El archivo `torino_simulation.py` permite simular los distintos algoritmos y observar los resultados.

