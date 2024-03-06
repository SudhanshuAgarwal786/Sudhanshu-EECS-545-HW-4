from builtins import range
import numpy as np
import math


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on the ipython notebook.
    """
    print("Hello from cnn_layers.py!")


def conv_forward(x, w):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_prime = H - HH + 1
    W_prime = W - WW + 1
    out = np.zeros((N, F, H_prime, W_prime))

    for n in range(N):
        for f in range(F):
            for j in range(H_prime):
                for i in range(W_prime):
                    out[n, f, j, i] = np.sum(x[n, :, j:j+HH, i:i+WW] * w[f, :, :, :])
    cache = (x, w)
    return out, cache




def conv_backward(dout, cache):
    x, w = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_prime = H - HH + 1
    W_prime = W - WW + 1

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)

    for n in range(N):
        for f in range(F):
            for j in range(H_prime):
                for i in range(W_prime):
                    dw[f, :, :, :] += x[n, :, j:j+HH, i:i+WW] * dout[n, f, j, i]
                    dx[n, :, j:j+HH, i:i+WW] += w[f, :, :, :] * dout[n, f, j, i]
    return dx, dw


def max_pool_forward(x, pool_param):
    N, C, H, W = x.shape
    p_H = pool_param['pool_height']
    p_W = pool_param['pool_width']
    stride = pool_param['stride']

    H_out = 1 + (H - p_H) // stride
    W_out = 1 + (W - p_W) // stride

    out = np.zeros((N, C, H_out, W_out))

    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h1 = h * stride
                    w1 = w * stride
                    out[n, c, h, w] = np.max(x[n, c, h1:h1+p_H, w1:w1+p_W])

    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    x, pool_param = cache
    N, C, H, W = x.shape
    p_H = pool_param['pool_height']
    p_W = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = 1 + (H - p_H) // stride
    W_out = 1 + (W - p_W) // stride

    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h1 = h * stride
                    w1 = w * stride
                    window = x[n, c, h1:h1+p_H, w1:w1+p_W]
                    mask = window == np.max(window)
                    dx[n, c, h1:h1+p_H, w1:w1+p_W] += dout[n, c, h, w] * mask

    return dx

