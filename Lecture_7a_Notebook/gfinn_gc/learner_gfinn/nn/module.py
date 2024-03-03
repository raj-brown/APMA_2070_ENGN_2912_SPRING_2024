"""
@author: jpzxshi
"""
import abc
import torch

# simply define a ReQUr function
def requr(input):
    '''
    Applies the Rectified Quadratic Unit regularization (ReQUr) function element-wise:

        ReQUr(x) = max(x**2, 0) - max((x-0.5)**2,0)
    '''
    return torch.relu(input**2) - torch.relu((input-0.5)**2) # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions

# simply define a ReQUr function
def requ(input):
    '''
    Applies the Rectified Quadratic Unit regularization (ReQUr) function element-wise:

        ReQU(x) = max(x**2, 0)
    '''
    return torch.relu(input**2)



class Module(torch.nn.Module):
    '''Standard module format. 
    '''
    def __init__(self):
        super(Module, self).__init__()
        self.activation = None
        self.initializer = None
        
        self.__device = None
        self.__dtype = None
        
    @property
    def device(self):
        return self.__device
        
    @property
    def dtype(self):
        return self.__dtype

    @device.setter
    def device(self, d):
        if d == 'cpu':
            self.cpu()
            for module in self.modules():
                if isinstance(module, Module):
                    module.__device = torch.device('cpu')
        elif d == 'gpu':
            self.cuda()
            for module in self.modules():
                if isinstance(module, Module):
                    module.__device = torch.device('cuda')
        else:
            raise ValueError
    
    @dtype.setter
    def dtype(self, d):
        if d == 'float':
            self.to(torch.float32)
            for module in self.modules():
                if isinstance(module, Module):
                    module.__dtype = torch.float32
        elif d == 'double':
            self.to(torch.float64)
            for module in self.modules():
                if isinstance(module, Module):
                    module.__dtype = torch.float64
        else:
            raise ValueError

    @property
    def act(self):
        if self.activation == 'sigmoid':
            return torch.sigmoid
        elif self.activation == 'relu':
            return torch.relu
        elif self.activation == 'tanh':
            return torch.tanh
        elif self.activation == 'elu':
            return torch.elu
        elif self.activation == 'sin':
            return torch.sin
        elif self.activation == 'cos':
            return torch.cos
        elif self.activation == 'silu':
            return torch.nn.SiLU()
        elif self.activation == 'requ':
            return requ
        elif self.activation == 'requr':
            return requr
        else:
            raise NotImplementedError
    
    @property        
    def Act(self):
        if self.activation == 'sigmoid':
            return torch.nn.Sigmoid()
        elif self.activation == 'relu':
            return torch.nn.ReLU()
        elif self.activation == 'tanh':
            return torch.nn.Tanh()
        elif self.activation == 'elu':
            return torch.nn.ELU()
        elif self.activation == 'sin':
            return torch.sin
        elif self.activation == 'silu':
            return torch.nn.SiLU()
        else:
            raise NotImplementedError

    @property
    def weight_init_(self):
        if self.initializer == 'He normal':
            return torch.nn.init.kaiming_normal_
        elif self.initializer == 'He uniform':
            return torch.nn.init.kaiming_uniform_
        elif self.initializer == 'Glorot normal':
            return torch.nn.init.xavier_normal_
        elif self.initializer == 'Glorot uniform':
            return torch.nn.init.xavier_uniform_
        elif self.initializer == 'orthogonal':
            return torch.nn.init.orthogonal_
        elif self.initializer == 'default':
            if self.activation == 'relu':
                return torch.nn.init.kaiming_normal_
            elif self.activation == 'tanh':
                return torch.nn.init.xavier_uniform_
            elif self.activation == 'silu':
                return torch.nn.init.xavier_uniform_
            # elif self.activation == 'sin':
            #     return torch.nn.init.xavier_uniform_
            else:
                return lambda x: None
        else:
            raise NotImplementedError
            
class StructureNN(Module):
    '''Structure-oriented neural network used as a general map based on designing architecture.
    '''
    def __init__(self):
        super(StructureNN, self).__init__()
        
    def predict(self, x, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        return self(x).cpu().detach().numpy() if returnnp else self(x)
    
class LossNN(Module, abc.ABC):
    '''Loss-oriented neural network used as an algorithm based on designing loss.
    '''
    def __init__(self):
        super(LossNN, self).__init__()
        
    #@final
    def forward(self, x):
        return x
    
    @abc.abstractmethod
    def criterion(self, X, y):
        pass
    
    @abc.abstractmethod
    def predict(self):
        pass

