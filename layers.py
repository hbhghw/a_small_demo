import tensorflow as tf

def identity(x):
    return tf.identity(x)

def relu6(x):
    return tf.nn.relu6(x)

def hardSigmoid(x):
    return relu6(x+3)/6

def hardSwish(x):
    return x*hardSigmoid(x)

def squeeze(x):
    x = tf.squeeze(x,1)
    x = tf.squeeze(x,1)
    return x

def globalAveragePooling(x):
    pool_size = x.shape[1:3]
    x = tf.layers.average_pooling2d(x,pool_size,strides=pool_size)
    return x

def batchNormalization(x):
    return tf.layers.batch_normalization(x,momentum=0.99)

def conv_bn_relu(x,filters,kernel_size=3,stride=1,padding='same',norm_layer=None,act_layer='relu',
                 use_bias=True,l2_reg = 1e-5):
    x = tf.layers.conv2d(x,filters,kernel_size,stride,padding='same',use_bias=use_bias)
    if norm_layer:
        x = batchNormalization(x)
    if act_layer == 'relu':
        x = tf.nn.relu(x)
    elif act_layer =='relu6':
        x = relu6(x)
    elif act_layer == 'hswish':
        x = hardSwish(x)
    elif act_layer == 'hsigmoid':
        x = hardSigmoid(x)
    elif act_layer=='sigmoid':
        x = tf.sigmoid(x)
    else:
        x = identity(x)

    return x

def bneck(inputs,out_channels,exp_channels,kernel_size,stride,use_se,act_layer,l2_reg=1e-5,index=None):
    in_channels = inputs.shape[-1]
    x = conv_bn_relu(inputs,exp_channels,kernel_size=1,norm_layer='bn',act_layer=act_layer,use_bias=False,l2_reg=l2_reg)
    w = tf.get_variable(f'dp_w{index}',shape=[kernel_size,kernel_size,x.shape[-1],1])
    depthwise = tf.nn.depthwise_conv2d(x,w,(1,stride,stride,1),padding='SAME')
    x = batchNormalization(depthwise)
    if use_se:
        x = seBottleneck(x,l2_reg=l2_reg)
    if act_layer=='relu':
        x = tf.nn.relu(x)
    elif act_layer=='hswish':
        x = hardSwish(x)
    else:
        x = identity(x)
    x = conv_bn_relu(x,out_channels,kernel_size=1,norm_layer='bn',act_layer=None,l2_reg=l2_reg)
    if stride==1 and in_channels == out_channels:
        return inputs + x
    else:
        return x

def seBottleneck(inputs,reduction=4,l2_reg=0.01):
    in_channels = inputs.shape[-1]
    x = globalAveragePooling(inputs)
    x = conv_bn_relu(x,in_channels//reduction,kernel_size=1,norm_layer=None,act_layer='relu',
                     use_bias=False,l2_reg=l2_reg)
    x = conv_bn_relu(x,in_channels,kernel_size=1,norm_layer='hsigmoid',use_bias=False,l2_reg=l2_reg)
    return x*inputs

def lastStage(x,penultimate_channels,last_channels,num_classes,l2_reg):
    x = conv_bn_relu(x,penultimate_channels,kernel_size=1,stride=1,norm_layer='bn',act_layer='hswish',
                     use_bias=False,l2_reg=l2_reg)
    x = globalAveragePooling(x)
    x = conv_bn_relu(x,last_channels,kernel_size=1,norm_layer=None,act_layer='hswish',l2_reg=l2_reg)
    x = tf.nn.dropout(x,rate=0.2)
    x = conv_bn_relu(x,num_classes,kernel_size=1,norm_layer=None,act_layer='softmax',l2_reg=l2_reg)
    x = squeeze(x)
    return x


