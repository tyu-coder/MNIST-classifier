from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None,initialize_method=None,dropout_p=None):
        if initialize_method is None:
            initialize_method=lambda size:np.random.randn(*size)*np.sqrt(2./size[0])
        self.size_list = size_list
        self.act_func = act_func
        self.lambda_list=lambda_list
        self.initialize_method=initialize_method
        self.layers=[]

        if size_list is not None and act_func is not None:
            
            for i in range(len(size_list) - 1):
                if lambda_list is not None:
                    layer = Linear(
                        in_dim=size_list[i], 
                        out_dim=size_list[i + 1],
                        initialize_method=self.initialize_method,
                        weight_decay=True,
                        weight_decay_lambda=lambda_list[i]
                    )
                else:
                    layer=Linear(
                        in_dim=size_list[i], 
                        out_dim=size_list[i + 1],
                        initialize_method=self.initialize_method

                    )
                self.layers.append(layer)   
                if i < len(size_list) - 2:
                    if act_func == 'Logistic':
                       raise NotImplementedError
                    elif act_func == 'ReLU':
                       self.layers.append(ReLU())
                       if dropout_p is not None:
                           drop=Dropout(p=dropout_p)
                           drop.training=True
                           self.layers.append(drop)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.layers = []

            
        for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.params['W'] =param_list[i + 2]['W']
                layer.params['b'] =param_list[i + 2]['b']
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W' : layer.params['W'], 
                    'b' : layer.params['b'], 
                    'weight_decay' : layer.weight_decay, 
                    'lambda' : layer.weight_decay_lambda
                })
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self,size_list=None,kernel_size_list=None,stride_list=None,padding_list=None,act_func=None,lambda_list=None,initialize_method=None):
        if initialize_method is None:
            initialize_method=lambda size: np.random.randn(*size)*np.sqrt(2./size[0])
        self.size_list=size_list
        self.kernel_size_list=kernel_size_list
        self.stride_list=stride_list
        self.padding_list=padding_list
        self.act_func=act_func
        self.lambda_list=lambda_list
        self.initialize_method=initialize_method
        self.layers=[]
        self.current_height=28
        self.current_width=28
        for i in range(len(size_list)-1):
            conv_layer=conv2D(
                in_channels=size_list[i],
                out_channels=size_list[i+1],
                kernel_size=kernel_size_list[i],
                stride=stride_list[i],
                padding=padding_list[i],
                initialize_method=self.initialize_method

            )
            self.layers.append(conv_layer)
            self.current_height=(self.current_height+2*padding_list[i]-kernel_size_list[i])//stride_list[i]+1
            self.current_width=(self.current_width+2*padding_list[i]-kernel_size_list[i])//stride_list[i]+1
            if act_func=='ReLU':
                self.layers.append(ReLU())

        final_channels=size_list[-1]
        self.flatten_size=final_channels*self.current_height*self.current_width
        linear_layer=Linear(
            in_dim=self.flatten_size,
            out_dim=10,
            initialize_method=self.initialize_method,
            weight_decay=lambda_list[-1] if lambda_list is not None else False
        )
        self.layers.append(linear_layer)

        

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        for layer in self.layers:
            if isinstance(layer,Linear):
                X=X.reshape(X.shape[0],-1)
            X=layer(X)
        return X

    def backward(self, loss_grad):
        grads=loss_grad
        for i in range(len(self.layers)-1,-1,-1):
            layer=self.layers[i]
            if isinstance(layer,Linear):
                grads=layer.backward(grads)
                if i>0 and isinstance(self.layers[i-1],ReLU):
                    batch_size=grads.shape[0]
                    grads=grads.reshape(batch_size,self.size_list[-1],self.current_height,self.current_width)
            else:
                grads=layer.backward(grads)
        return grads

    
    def load_model(self, param_list):
        pass
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.kernel_size_list,self.stride_list,self.padding_list,self.act_func]
        for layer in self.layers:
            if isinstance(layer,conv2D):
                param_list.append({
                    'W' : layer.params['W'], 
                    'b' : layer.params['b'], 
                    'kernel_size':layer.kernel_size,
                    'stride':layer.stride,
                    'padding':layer.padding
                })
        if isinstance(self.layers[-1],Linear):
                linear_layer=self.layers[-1]
                param_list.append({
                    'W' : linear_layer.params['W'], 
                    'b' : linear_layer.params['b']
            })                
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)        