import numpy as np


class Tensor:
    '''
    terminal nodes will be placeholder,variable,constant,
    and tensor refers to the output of an operation node
    '''

    def __init__(self, dtype=None, shape=None, name=None):
        self.shape = shape
        self.dtype = dtype
        self.inputs = None
        self.name = name
        self.operation = None

    def compute_output(self):
        '''
        compute output shape for high dimension tensors' calculation
        :return:
        '''
        raise NotImplementedError

    def forward(self):
        pass

    def backward(self, upstream_grade):  # backward
        if isinstance(self, Placeholder) or isinstance(self, Constant):
            return
        if isinstance(self, Variable):
            self.update(upstream_grade)
            return
        gradients = self.operation.compute_gradient(self.inputs)
        for i, tensor in enumerate(self.inputs):
            tensor.backward(upstream_grade * gradients[i])

    def __add__(self, other):
        '''
        a tensor A add other tensor B will return a tensor that takes [A,B] as it's inputs
        C = A+B equals to C = add(A,B)
        '''
        new_tensor = Tensor()
        other = convert_to_tensor(other)
        new_tensor.inputs = [self, other]
        new_tensor.operation = add
        return new_tensor

    __radd__ = __add__

    def __mul__(self, other):
        # C = A * B , C = multiply(A,B)
        new_tensor = Tensor()
        other = convert_to_tensor(other)
        new_tensor.inputs = [self, other]
        new_tensor.operation = multiply
        return new_tensor

    __rmul__ = __mul__

    def __sub__(self, other):
        # C = self-A ,also C = sub(self,A)
        new_tensor = Tensor()
        other = convert_to_tensor(other)
        new_tensor.inputs = [self, other]
        new_tensor.operation = sub
        return new_tensor

    def __rsub__(self, other):
        # C = A-self ,also C = sub(A,self)
        new_tensor = Tensor()
        other = convert_to_tensor(other)
        new_tensor.inputs = [other, self]  # here other-self
        new_tensor.operation = sub
        return new_tensor

    def __neg__(self):
        # -A
        # return 0-self
        new_tensor = Tensor()
        new_tensor.inputs = [self]
        new_tensor.operation = sub
        return new_tensor

    def eval(self):
        '''
        note : placeholder,variable,constant nodes' eval method will return their value,
        tensor object will compute output recursively
        >>> example:
        X = Placeholder()
        W = Variable()
        B = Variable()
        C = X*W
        D = C + B
        D.eval() = add.compute([C,B]) = C.eval()+B.eval() = mul.compute([X,W])+B.eval() = X.eval() * W.eval() + B.eval()
        :return: eval value
        '''
        return self.operation.compute(self.inputs)


def convert_to_tensor(obj):
    '''
    convert an object to a Tensor object,so it will have Tensor' methods like eval() etc.
    :param obj: an object
    :return: a Tensor object
    '''
    if isinstance(obj, Tensor):
        return obj
    if isinstance(obj, int) or isinstance(obj, float):  # scaler
        return Constant(init_value=obj, dtype=type(obj), shape=[])
    raise (type(obj), "object cannot be converted to a tensor.")


class Placeholder(Tensor):
    '''
    Placeholder object should use feed() method each time before run
    '''

    def __init__(self, dtype=None, shape=None, name=None):
        super(Placeholder, self).__init__()
        self.shape = shape
        self.dtype = dtype
        self.name = name

    def feed(self, tensor):
        tensor = convert_to_tensor(tensor)
        self.value = tensor.value

    def eval(self):
        return self.value


class Variable(Tensor):
    def __init__(self, dtype=None, shape=None, init_value=None, name=None):
        super(Variable, self).__init__()
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.value = init_value

    def update(self, up_grad):
        self.value = self.value - up_grad

    def eval(self):
        return self.value


class Constant(Tensor):
    def __init__(self, dtype=None, shape=None, init_value=None, name=None):
        super(Constant, self).__init__()
        if init_value is None:
            raise ("constant tensor should be inited with a value")
        self.dtype = dtype
        self.shape = shape
        self.value = init_value

    def gradient(self):
        pass

    def eval(self):
        return self.value


class Operation:
    def __init__(self):
        self.inputs = None

    def compute_gradient(self, inputs):
        pass


class AddOperation(Operation):
    def __init__(self):
        super(AddOperation, self).__init__()

    def __call__(self, input1, input2):
        input1, input2 = convert_to_tensor(input1), convert_to_tensor(input2)
        new_tensor = Tensor()
        new_tensor.inputs = [input1, input2]
        new_tensor.operation = self
        return new_tensor

    def compute(self, inputs):
        return inputs[0].eval() + inputs[1].eval()

    def compute_gradient(self, inputs):
        return [1 for _ in inputs]


add = AddOperation()


class SubOperation(Operation):
    def __init__(self):
        super(SubOperation, self).__init__()

    def __call__(self, input1, input2):
        input1, input2 = convert_to_tensor(input1), convert_to_tensor(input2)
        new_tensor = Tensor()
        new_tensor.inputs = [input1, input2]
        new_tensor.operation = self
        return new_tensor

    def compute(self, inputs):
        return inputs[0].eval() - inputs[1].eval()

    def compute_gradient(self, inputs):
        return [1, -1]


sub = SubOperation()


class NegOperation(Operation):
    def __init__(self):
        super(NegOperation, self).__init__()

    def __call__(self, input1):
        input1 = convert_to_tensor(input1)
        new_tensor = Tensor()
        new_tensor.inputs = [input1]
        new_tensor.operation = self
        return new_tensor

    def eval(self, inputs):
        return inputs.value

    def compute_gradient(self, inputs):
        return [-1]


neg = NegOperation()


class MulOperation(Operation):
    def __init__(self):
        super(MulOperation, self).__init__()

    def __call__(self, input1, input2):
        input1, input2 = convert_to_tensor(input1), convert_to_tensor(input2)
        new_tensor = Tensor()
        new_tensor.inputs = [input1, input2]
        new_tensor.operation = self
        return new_tensor

    def compute(self, inputs):
        return inputs[0].eval() * inputs[1].eval()

    def compute_gradient(self, inputs):
        return [inputs[1].eval(), inputs[0].eval()]


multiply = MulOperation()


class ConvolutionOperation(Operation):
    def __init__(self):
        super(ConvolutionOperation, self).__init__()

    def __call__(self, inputs):
        inputs = convert_to_tensor(inputs)
        new_tensor = Tensor()
        self.compute_output_shape()
        return new_tensor

    def compute_output_shape(self):
        pass

    def forward(self):
        pass

    def eval(self):
        pass


Conv2D = ConvolutionOperation()


