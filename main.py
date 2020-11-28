import pandas as pd
import numpy as np
import random

import pandas as pd
import numpy as np


######################################################################################
######################################################################################
##                               Clases para ANN                                    ##
######################################################################################
######################################################################################

"""
Clase que representa una capa de la red
Contiene la matriz de pesos para cada conexion y el vector de bias.

"""
class Capa:
    def __init__(self, num_neuronas : int, cant_conn : int, activation = None, entrada = 3, ocultas = 1):
        self.num_neuronas = num_neuronas
        self.activation = activation
        self.b = np.random.rand(1, num_neuronas) * 2 - 1
        self.w = np.random.rand(cant_conn, num_neuronas) *2 - 1 

    def custom_rand(entrada, ocultas):
        r = (6/(entrada + ocultas)) ** (0.5)
        return (np.random.uniform(low=ocultas, high=entrada) * 2 * r - r)
"""
    Clase que contiene los datos y métodos para crear la red neuronal
    Contiene una lista con las capas y métodos para predecir y calcular el coste
"""
class Red:
    def __init__(self, shape, activation):
        self.capas = []
        for i, dim in enumerate(shape[:-1]):
            layer = Capa(num_neuronas=shape[i+1], cant_conn = shape[i], activation=activation)
            self.capas.append(layer)
    def forward_pass(self, X):
        out = [(None, X)]
        for l, capa in enumerate(self.capas):
            z = out[-1][1] @ self.capas[l].w + self.capas[l].b
            a = self.capas[l].activation(z)
            out.append((z, a)) 
        return out[-1]
    def evaluate(self, X, Ye):
        return np.mean((np.array(Ye) - np.array(self.forward_pass(X)[1])) ** 2)

    def get_weights(self):
        return [ l.w for l in self.capas ]

    def set_weights(self, weights):
        for i in range(len(self.capas)):
            self.capas[i].w = weights[i]
 
"""
    Función de activación customizada, en este caso es la del mexican hat
"""
def custom_activation(x):
    return x * np.exp(-0.5 * np.power(x, 2))


######################################################################################
######################################################################################
##                               Clases para PSO                                    ##
######################################################################################
######################################################################################
BIG_SCORE = 1.e6

class Particle:
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.init_weights = model.get_weights()
        self.velocities = [None] * len(self.init_weights)
        self.length = len(self.init_weights)
        for i, layer in enumerate(self.init_weights):
            self.velocities[i] = np.random.rand(*layer.shape) / 5 - 0.10

        self.best_weights = None
        self.best_score = BIG_SCORE

    def get_score(self, x, y, update=True):
        local_score = self.model.evaluate(x, y)
        if local_score < self.best_score and update:
            self.best_score = local_score
            self.best_weights = self.model.get_weights()

        return local_score

    def _update_velocities(self, global_best_weights, depth):
        new_velocities = [None] * len(self.init_weights)
        weights = self.model.get_weights()
        local_rand, global_rand = random.random(), random.random()

        for i, layer in enumerate(weights):
            if i >= depth:
                new_velocities[i] = self.velocities[i]
                continue
            new_v = self.params['acc'] * self.velocities[i]
            new_v = new_v + self.params['local_acc'] * local_rand * (self.best_weights[i] - layer)
            new_v = new_v + self.params['global_acc'] * global_rand * (global_best_weights[i] - layer)
            new_velocities[i] = new_v

        self.velocities = new_velocities

    def _update_weights(self, depth):
        old_weights = self.model.get_weights()
        new_weights = [None] * len(old_weights)
        for i, layer in enumerate(old_weights):
            if i>= depth:
                new_weights[i] = layer
                continue
            new_w = layer + self.velocities[i]
            new_weights[i] = new_w

        self.model.set_weights(new_weights)

    def step(self, x, y, global_best_weights,depth=None):
        if depth is None:
            depth = self.length
        self._update_velocities(global_best_weights, depth)
        self._update_weights(depth)
        return self.get_score(x, y)

    def get_best_weights(self):
        return self.best_weights






class Optimizer:
    def __init__(self, model, params_red,
                 n=10,
                 acceleration=0.1,
                 local_rate=1.0,
                 global_rate=1.0):

        self.n_particles = n
        self.structure = params_red
        self.particles = [None] * n
        self.length = len(model.get_weights())

        params_pso = {'acc': acceleration, 'local_acc': local_rate, 'global_acc': global_rate}

        for i in range(n-1):
            m = Red(self.structure['shape'], self.structure['activation'])
            self.particles[i] = Particle(m, params_pso)

        self.particles[n-1] = Particle(m, params_pso)

        self.global_best_weights = None
        self.global_best_score = BIG_SCORE
    
    def fit(self, x, y, steps=0, batch_size=32):
        num_batches = len(x) // batch_size


        for i, p in enumerate(self.particles):
            local_score = p.get_score(x, y)

            if local_score < self.global_best_score:
                self.global_best_score = local_score
                self.global_best_weights = p.get_best_weights()

        print("PSO -- Initial best score {:0.4f}".format(self.global_best_score))


        for i in range(steps):
            #for j in range(num_batches):
            #    x_ = x[j*batch_size:(j+1)*batch_size,:]
            #    y_ = y[j*batch_size:(j+1)*batch_size]
#
            for p in self.particles:
                local_score = p.step(x, y, self.global_best_weights)

                if local_score < self.global_best_score:
                    self.global_best_score = local_score
                    self.global_best_weights = p.get_best_weights()

    def get_best_model(self):
        best_model = Red(self.structure['shape'], self.structure['activation'])
        best_model.set_weights(self.global_best_weights)
        return best_model








######################################################################################
######################################################################################
##                        Ejecución del Programa                                    ##
######################################################################################
######################################################################################

def main():
    p = 2 # numero de valores de entrada
    shape = [p, 3, 1]
    N = 100 # Numero de particulas
    acc = 0.1 # Aceleración de las partículas
    lr = 1 # global y local rate
    gr = 1


    params = { 'shape': shape , 'activation' : custom_activation }
    red = Red(shape, custom_activation)

    pso = Optimizer(red, params, n=N, acceleration=acc, local_rate=lr, global_rate=gr)
    x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [[0], [1], [1], [0]]
    BATCH_SIZE = 1
    STEPS = 1000
    pso.fit(x_train, y_train, steps=STEPS, batch_size=BATCH_SIZE)
    model_p = pso.get_best_model()
    print(model_p.evaluate(x_train, y_train))

    



if __name__ == "__main__":
    main()