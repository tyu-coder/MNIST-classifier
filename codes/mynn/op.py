from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.optimizable=True
        self.params={
            'W' : initialize_method(size=(in_dim, out_dim)),
            'b' : initialize_method(size=(1, out_dim))
        }        
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
                
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """ 
        self.input = X
        return X@self.params['W']+self.params['b']

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size=self.input.shape[0]
        self.grads['W']=self.input.T@grad/batch_size
        self.grads['b']=np.sum(grad,axis=0,keepdims=True)/batch_size
        if self.weight_decay:
            self.grads['W']+=self.weight_decay_lambda*self.params['W']
        return grad@self.params['W'].T   
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=None) -> None:
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        if initialize_method is None:
            initialize_method=lambda size:np.random.randn(*size)*np.sqrt(2./size[0])
        self.initialize_method=initialize_method

        self.params={
            'W':self.initialize_method((out_channels,in_channels,kernel_size,kernel_size)),
            'b':np.zeros((out_channels,),dtype=np.float64)
        }
        self.grads={}

    def __call__(self, X):
        return self.forward(X)
    def im2col(self,X,field_height,field_width,padding,stride):
        N,C,H,W=X.shape
        out_height=(H+2*padding-field_height)//stride+1
        out_width=(W+2*padding-field_width)//stride+1
        
        i0=np.repeat(np.arange(field_height),field_width)
        i0=np.tile(i0,C)
        i1=stride*np.repeat(np.arange(out_height),out_width)

        j0=np.repeat(np.arange(field_width),field_height)
        j0=np.tile(j0,C)
        j1=stride*np.repeat(np.arange(out_width),out_height)

        i=i0.reshape(-1,1)+i1.reshape(1,-1)
        j=j0.reshape(-1,1)+j1.reshape(1,-1)

        k=np.repeat(np.arange(C),field_height*field_width).reshape(-1,1)

        X_padded=np.pad(X,((0,0),(0,0),(padding,padding),(padding,padding)),mode='constant')
        cols=X_padded[:,k,i,j]
        cols=cols.transpose(1,2,0).reshape(field_height*field_width*C,-1)
        return cols
    def col2im(self,cols,X_shape,field_height,field_width,padding,stride):
        N,C,H,W=X_shape
        H_padded,W_padded=H+2*padding,W+2*padding
        X_padded=np.zeros((N,C,H_padded,W_padded),dtype=cols.dtype)
        out_height=(H+2*padding-field_height)//stride+1
        out_width=(W+2*padding-field_width)//stride+1
        
        i0=np.repeat(np.arange(field_height),field_width)
        i0=np.tile(i0,C)
        i1=stride*np.repeat(np.arange(out_height),out_width)

        j0=np.tile(np.arange(field_width),field_height)
        j0=np.tile(j0,C)
        j1=stride*np.tile(np.arange(out_width),out_height)

        i=i0.reshape(-1,1)+i1.reshape(1,-1)
        j=j0.reshape(-1,1)+j1.reshape(1,-1)

        k=np.repeat(np.arange(C),field_height*field_width).reshape(-1,1)

        cols_reshaped=cols.reshape(C*field_height*field_width,-1,N)
        cols_reshaped=cols_reshaped.transpose(2,0,1)
        np.add.at(X_padded,(slice(None),k,i,j),cols_reshaped)
        if padding==0:
            return X_padded
        return X_padded[:,:,padding:-padding,padding:-padding]
    
    def forward(self, X):
        """
        input X: [batch, channels, H, 
        W : [1, out, in, k, k]
        no padding
        """
        self.X=X
        N,C,H,W=X.shape
        self.X_cols=self.im2col(X,self.kernel_size,self.kernel_size,self.padding,self.stride)
        W_col=self.params['W'].reshape(self.out_channels,-1)
        out=W_col@self.X_cols+self.params['b'].reshape(-1,1)
        out_height=(H+2*self.padding-self.kernel_size)//self.stride+1
        out_width=(W+2*self.padding-self.kernel_size)//self.stride+1
        out=out.reshape(self.out_channels,out_height,out_width,N)
        out=out.transpose(3,0,1,2)
        return out


    def backward(self, dout):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        N,C,H,W=self.X.shape
        dout_reshaped=dout.transpose(1,2,3,0).reshape(self.out_channels,-1)
        self.grads['W']=dout_reshaped@self.X_cols.T
        self.grads['W']=self.grads['W'].reshape(self.params['W'].shape)
        self.grads['b']=np.sum(dout,axis=(0,2,3))
        W_flat=self.params['W'].reshape(self.out_channels,-1)
        dX_cols=W_flat.T@dout_reshaped
        dX=self.col2im(dX_cols,self.X.shape,self.kernel_size,self.kernel_size,self.padding,self.stride)
        return dX
        pass
    
    def clear_grad(self):
        self.grads = {'W' : np.zeros_like(self.params['W']), 'b' : np.zeros_like(self.params['b'])}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        grad_relu=grads*(self.input>0)
        return grad_relu

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model=model
        self.max_classes=max_classes
        self.has_softmax=True

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        self.predicts=predicts
        self.labels=labels
        self.softmax=softmax(predicts)
        batch_size=predicts.shape[0]
        one_hot=np.zeros_like(self.softmax)
        one_hot[np.arange(batch_size),labels]=1
        log_softmax=np.log(self.softmax+1e-9)
        loss=-np.sum(one_hot*log_softmax)/batch_size
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        # Then send the grads to model for back propagation
        batch_size=self.labels.shape[0]
        one_hot=np.zeros_like(self.softmax)
        one_hot[np.arange(batch_size),self.labels]=1
        self.grads=(self.softmax-one_hot)/batch_size
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, model = None, lambda_reg=0.01) -> None:
        super().__init__()
        self.model=model
        self.lambda_reg=lambda_reg
    def forward(self):
        l2_loss=0
        for layer in self.model.layers:
            if hasattr(layer,'params')and 'W' in layer.params:
                       l2_loss+=np.sum(layer.params['W']**2)
        return self.lambda_reg*l2_loss
    def backward(self):
        for layer in self.model.layers:
            if hasattr(layer,'grads' )and 'W' in layer.grads:
                layer.grads['W']+=2*self.lambda_reg*layer.params['W']
        return self
    
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p=p
        self.mask=None
        self.optimizable=False

    def __call__(self,X):
        return self.forward(X)
    def forward(self,X):
        if hasattr(self,'training')and self.training:
            self.mask=(np.random.rand(*X.shape)>self.p).astype(X.dtype)
            self.mask/=(1.0-self.p)
            return X*self.mask
        else:
            self.mask=None
            return X
    def backward(self,grad):
        if self.mask is not None:
            return grad*self.mask
        else:
            return grad