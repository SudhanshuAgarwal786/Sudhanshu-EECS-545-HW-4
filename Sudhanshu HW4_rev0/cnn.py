import numpy as np

from layers import relu_forward, fc_forward, fc_backward, relu_backward, softmax_loss
from cnn_layers import conv_forward, conv_backward, max_pool_forward, max_pool_backward


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on the ipython notebook.
    """
    print("Hello from cnn.py!")


class ConvNet(object):
    def __init__(self, input_dim=(1, 28, 28), num_filters_1=6, num_filters_2=16, filter_size=5,
                 hidden_dim=100, num_classes=10, dtype=np.float32):
        self.params = {}
        self.dtype = dtype
        C, H, W = input_dim

        # Convolutional layer weights
        k1 = 1 / (C * filter_size**2)
        self.params['W1'] = np.random.uniform(-np.sqrt(k1), np.sqrt(k1), (num_filters_1, C, filter_size, filter_size))
        self.params['W2'] = np.random.uniform(-np.sqrt(k1), np.sqrt(k1), (num_filters_2, num_filters_1, filter_size, filter_size))

        # Calculate output size after two convolutions and pooling layers
        conv_out_h = (H - filter_size + 1) // 2
        conv_out_w = (W - filter_size + 1) // 2
        conv_out_h2 = (conv_out_h - filter_size + 1) // 2
        conv_out_w2 = (conv_out_w - filter_size + 1) // 2
        pool_out_size = num_filters_2 * conv_out_h2 * conv_out_w2

        # Fully connected layer weights
        k2 = 1 / pool_out_size
        self.params['W3'] = np.random.uniform(-np.sqrt(k2), np.sqrt(k2), (pool_out_size, hidden_dim))
        self.params['b3'] = np.zeros(hidden_dim)

        k3 = 1 / hidden_dim
        self.params['W4'] = np.random.uniform(-np.sqrt(k3), np.sqrt(k3), (hidden_dim, num_classes))
        self.params['b4'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        W1, W2 = self.params['W1'], self.params['W2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        # Forward pass
        conv_out_1, cache_conv_1 = conv_forward(X, W1)
        relu_out_1, cache_relu_1 = relu_forward(conv_out_1)
        pool_out_1, cache_pool_1 = max_pool_forward(relu_out_1, pool_param)
        
        conv_out_2, cache_conv_2 = conv_forward(pool_out_1, W2)
        relu_out_2, cache_relu_2 = relu_forward(conv_out_2)
        pool_out_2, cache_pool_2 = max_pool_forward(relu_out_2, pool_param)
        
        # Flatten the output for the FC layer
        fc_input = pool_out_2.reshape(pool_out_2.shape[0], -1)
        fc_out_1, cache_fc_1 = fc_forward(fc_input, W3, b3)
        relu_out_3, cache_relu_3 = relu_forward(fc_out_1)
        scores, cache_fc_2 = fc_forward(relu_out_3, W4, b4)
        
        if y is None:
            return scores
        
        # Compute loss
        loss, dscores = softmax_loss(scores, y)
        
        # Backward pass
        grads = {}
        dx4, dw4, db4 = fc_backward(dscores, cache_fc_2)
        grads['W4'], grads['b4'] = dw4, db4
        
        drelu3 = relu_backward(dx4, cache_relu_3)
        dx3, dw3, db3 = fc_backward(drelu3, cache_fc_1)
        grads['W3'], grads['b3'] = dw3, db3
        
        dpool2 = dx3.reshape(pool_out_2.shape)
        drelu2 = max_pool_backward(dpool2, cache_pool_2)
        dconv2 = relu_backward(drelu2, cache_relu_2)
        dx2, dw2 = conv_backward(dconv2, cache_conv_2)
        grads['W2'] = dw2
        
        dpool1 = max_pool_backward(dx2, cache_pool_1)
        drelu1 = relu_backward(dpool1, cache_relu_1)
        _, dw1 = conv_backward(drelu1, cache_conv_1)
        grads['W1'] = dw1
        
        return loss, grads
