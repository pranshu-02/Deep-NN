#libraries
import numpy as np
import h5py
import math

def sigmoid(Z):
    """
    Implements the sigmoid activation.
    """

    A = 1.0/(1.0+np.exp(-Z))
    cache = Z

    return A, cache

def relu(Z):
    """
    Implement the RELU function.
    """

    A = np.maximum(0,Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit
    """

    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit
    """

    Z = cache
    s,_ = sigmoid(Z)
    dZ = dA * s * (1-s)
    return dZ

def initialize_parameters(layers_dims):

    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros(shape=(layers_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)

    return A, cache



def forward_propagation(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    """

    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation='relu')
        caches.append(cache)

    AL, cache = linear_activation_forward(A,parameters['W' + str(L)],parameters['b' + str(L)], activation='sigmoid')
    caches.append(cache)
    return AL, caches



def compute_cost(AL, Y, parameters ,lambd):
    """
    Implement the cost function.
    """
    L = len(parameters) // 2
    m = Y.shape[1]
    cost = -1 / m * np.sum(np.nan_to_num(Y * np.log(AL) + (1-Y) * np.log(1-AL)))
    cost+= 0.5*(lambd/m)*sum(np.linalg.norm(parameters['W' + str(i)])**2 for i in range(1,L))
    return cost



def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * dZ @ A_prev.T 
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T @ dZ

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db



def backward_propagation(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    """
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')
 
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

def mini_batches(X, Y, mini_batch_size = 64):
    """
    Create random minibatches from (X, Y)
    """
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((10,m))

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        end = m - mini_batch_size * math.floor(m / mini_batch_size)
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def initialize_adam(parameters) :
    """
    Initializes v and s for Adam optimizer
    """

    L = len(parameters) // 2
    v = {}
    s = {}
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

        s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l + 1)])

    return v, s



def update_parameters_with_adam(parameters, grads, v, s, t, lambd, learning_rate, mini_batch_size, beta1, beta2, epsilon):
    """
    Update parameters using Adam optimizer
    """

    L = len(parameters) // 2                 
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))

        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))
        parameters["W" + str(l + 1)] = (1-learning_rate*(lambd/mini_batch_size))*parameters["W" + str(l + 1)]        
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon)


    return parameters, v, s



def NeuralNet(X, Y, X1, Y1, layer_dims, mini_batch_size = 128, num_epochs = 1000, learning_rate = 0.001, lambd = 0.5 ,beta1= 0.9 ,beta2 = 0.999 ,epsilon =1e-8 ):
    """Implements The Complete Deep Neural Network."""
    test_set_len = len(Y1)
    t = 0
    parameters = initialize_parameters(layer_dims)
    v, s = initialize_adam(parameters)

    for i in range(num_epochs):
        learning_rate = (1-(i/100))*learning_rate
        minibatches = mini_batches(X, Y, mini_batch_size)
        for minibatch in minibatches:
            (mini_X, mini_Y) = minibatch
            AL, caches = forward_propagation(mini_X, parameters)
            grads = backward_propagation(AL, mini_Y, caches)
            t = t + 1
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, lambd, learning_rate, mini_batch_size, beta1, beta2,  epsilon )
        cost = compute_cost(forward_propagation(X, parameters)[0], Y, parameters ,lambd)
        print("Cost after epoch %i: %f" % (i, np.sum(cost)))
        results = list(zip(np.argmax(forward_propagation(X1 , parameters)[0],axis = 0),Y1))
        p = sum(int(x == y) for (x, y) in results)
        print("Accuracy Is ",p/test_set_len)
