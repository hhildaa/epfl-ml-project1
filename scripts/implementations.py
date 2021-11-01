import numpy as np

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>

    Input:
    y: prediction variable
    tx: input data
    batch_size: size of batches
    num_batches: number of batches to return from tx
    shuffle: boolean whether to shuffle dataset before batching

    Returns:
    Iterator over batches
    """
    
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_loss(y, tx, w, type="mse"):
    """
    Calculate the loss using mse or mae.
    
    Input:
    y: prediction variable
    tx: data
    w: weights (model)
    type: one of "mse" or "mae" for resp. loss function

    returns:
    loss
    """
    N = y.shape[0]
    fw = np.matmul(tx,w)
    if type=="mse":
        return (1/(2*N)*np.matmul(y-fw,(y-fw).T))
    if type=="mae":
        return (1/N*np.sum(np.abs(y-fw)))

def compute_gradient(y, tx, w):
    """
    Compute the gradient for least squares linear regression.

    Input:
    y: prediction variable
    tx: data
    w: current weights (model)

    returns:
    gradient
    """
    N=y.size
    e=y-np.matmul(tx,w)
    return(-1/N*np.matmul(tx.T,e))

def compute_stoch_gradient(y, tx, w):
    """
    Compute a stochastic gradient from just few examples n and their corresponding y_n labels.
    
    Input:
    y: prediction variable
    tx: data
    w: current weights (model)

    returns:
    (stochastic) gradient
    """
    N=y.size
    e=y-np.matmul(tx,w)
    return(-1/N*np.matmul(tx.T,e))

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent

    Input:
    y: prediction variable
    tx: data
    initial_w: initial weights used
    max_iters: number of iterations after which training is stopped
    gamma: step size for gradient descent

    Returns:
    w: final weights after max_iters updates
    loss: final loss of w*tx on y (training error)
    """
    w = initial_w
    for n_iter in range(max_iters):
        gr=compute_gradient(y,tx,w)
        loss=compute_loss(y,tx,w)
        w=w-gamma*gr
    return (w, loss)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent
    
    Input:
    y: prediction variable
    tx: data
    initial_w: initial weights used
    max_iters: number of iterations after which training is stopped
    gamma: step size for (stochastic) gradient descent

    Returns:
    w: final weights after max_iters updates
    loss: final loss of w*tx on y (training error)
    """
    batch_size = 1
    num_batches = 1 
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches):
            gr=compute_stoch_gradient(minibatch_y,minibatch_tx,w)
            w=w-gamma*gr
        loss=compute_loss(y,tx, w)
    return (w, loss)


def least_squares(y, tx):
    """
    Least squares linear regression using normal equations.
    
    Input:
    y: prediction variable
    tx: data

    Returns:
    w: weights of least squares linear regression
    mse: loss of w*tx on y (MSE, training error)
    """
    tx_t = np.transpose(tx)
    weights = np.linalg.solve(tx_t.dot(tx), tx_t.dot(y))
    mse = compute_loss(y, tx, weights)

    return weights, mse


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.
    
    Input:
    y: prediction variable
    tx: data
    lambda_: regularization parameter

    Returns:
    w: weights of ridge regression
    mse: loss of w*tx on y (MSE, training error)
    """
    tx_t = np.transpose(tx)
    N, D = tx.shape
    linear_func = tx_t.dot(tx) + (2 * N * lambda_ * np.identity(D))
    Xty = tx_t.dot(y)
    weights = np.linalg.solve(linear_func, Xty)
    mse = compute_loss(y, tx, weights)
    return weights, mse

def compute_ridge_gradient(y, tx, lambda_, w):
    """
    Computes gradient for ridge regression
    
    Input:
    y: prediction variable
    tx: data
    lambda_: regularization parameter
    w: current weights (model)

    returns:
    gradient
    """
    grad = compute_gradient(y, tx, w)
    return grad + 2*lambda_*w
            
def sigmoid(z):
    """Computes sigmoid function of z"""
    return 1.0 / (1 + np.exp(-z))


def clipping(h):
    """Clips value in h to some (positive and negative) threshold to avoid large numbers"""
    abs_threshold = 15
    h[h > abs_threshold] = abs_threshold
    h[h < -abs_threshold] = -abs_threshold
    return h


def cross_entropy_loss(y, tx, w):
    """
    Calculates cross entropy loss.

    Input:
    y: prediction variable
    tx: data
    w: current weights (model)

    returns:
    loss
    """
    # clipping values to avoid INF loss
    h = sigmoid(clipping(tx @ w))
    return np.squeeze((-y.T @ np.log(h)) + (-(1 - y).T @ np.log(1 - h)))
                  
    
def cross_entropy_gradient(y, tx, w):
    """
    Computes gradient for cross entropy loss
    
    Input:
    y: prediction variable
    tx: data
    w: current weights (model)

    returns:
    gradient
    """
    h = sigmoid(tx @ w)
    return (tx.T @ (h - y))


def regularized_cross_entropy_loss(y, tx, w, lambda_):
    """
    Calculates regularized cross entropy loss.

    Input:
    y: prediction variable
    tx: data
    w: current weights (model)
    lambda_: regularization parameter

    returns:
    loss
    """
    return cross_entropy_loss(y, tx, w) + lambda_ * np.squeeze(w.T @ w)


def regularized_cross_entropy_gradient(y, tx, w, lambda_):
    """
    Computes gradient for cross entropy loss
    
    Input:
    y: prediction variable
    tx: data
    w: current weights (model)
    lambda_: regularization parameter

    returns:
    gradient
    """
    return cross_entropy_gradient(y, tx, w) + 2 * lambda_ * w


def logistic_regression(y, tx, initial_w, max_iters, gamma, batch_size=1, num_batches=1):
    """
    Logistic regression using gradient descent or SGD
    
    Input:
    y: prediction variable
    tx: data
    initial_w: initial weights
    max_iters: number of iterations after which training is stopped
    gamma: step size for gradient descent
    batch_size: size of batches (size of tx for gradient descent)
    num_batches: number of batches to compute out of tx

    Returns:
    w: final weights after max_iters updates
    loss: final loss of w*tx on y (training error)
    """
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches):
            gr = cross_entropy_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * gr
            
        loss = cross_entropy_loss(y, tx, w)
    return (w, loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, batch_size=1, num_batches=1):
    """
    Regularized logistic regression using gradient descent or SGD
    
    Input:
    y: prediction variable
    tx: data
    lambda_: regularization parameter
    initial_w: initial weights used
    max_iters: number of iterations after which training is stopped
    gamma: step size for gradient descent
    batch_size: size of batches (size of tx for gradient descent)
    num_batches: number of batches to compute out of tx

    Returns:
    w: final weights after max_iters updates
    loss: final loss of w*tx on y (training error)
    """
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches):
            gr = regularized_cross_entropy_gradient(minibatch_y, minibatch_tx, w, lambda_)
            w = w - gamma * gr
            
        loss = regularized_cross_entropy_loss(y, tx, w, lambda_)
    return (w, loss)