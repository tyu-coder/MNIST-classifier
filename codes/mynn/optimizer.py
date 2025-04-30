from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    grad=layer.grads[key]

                    '''if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)'''
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]
                layer.clear_grad()

class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu=mu
        self.momentum={}
        for layer in self.model.layers:
            if layer.optimizable:
                self.momentum[layer]={key:np.zeros_like(layer.params[key]) for key in layer.params}
                    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    self.momentum[layer][key]=self.mu*self.momentum[layer][key] - self.init_lr * layer.grads[key]
                    layer.params[key] +=self.momentum[layer][key] 
                layer.clear_grad()