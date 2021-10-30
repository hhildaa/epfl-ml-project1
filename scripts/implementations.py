import numpy as np

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
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
    """Calculate the loss using mse or mae."""
    N = y.shape[0]
    fw = np.matmul(tx,w)
    if type=="mse":
        return (1/(2*N)*np.matmul(y-fw,(y-fw).T))
    if type=="mae":
        return (1/N*np.sum(np.abs(y-fw)))

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N=y.size
    e=y-np.matmul(tx,w)
    return(-1/N*np.matmul(tx.T,e))

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    N=y.size
    e=y-np.matmul(tx,w)
    return(-1/N*np.matmul(tx.T,e))

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    w = initial_w
    for n_iter in range(max_iters):
        gr=compute_gradient(y,tx,w)
        loss=(compute_loss(y,tx,w))
        w=w-gamma*gr
#        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
#              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return (w, loss)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    batch_size = 1
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gr=compute_stoch_gradient(minibatch_y,minibatch_tx,w)
            w=w-gamma*gr
        loss=compute_loss(y,tx, w)
            
#        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
#              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return (w, loss)


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    tx_t = np.transpose(tx)
    N, D = tx.shape
    linear_func = tx_t.dot(tx) + (2 * N * lambda_ * np.identity(D))
    Xty = tx_t.dot(y)
    weights = np.linalg.solve(linear_func, Xty)
    mse = compute_loss(y, tx, weights)
    return weights, mse

def compute_ridge_gradient(y, tx, lambda_, w):
    """returns gradient for ridge regression"""
    grad = compute_gradient(y, tx, w)
    return grad + 2*lambda_*w
            
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def clipping(h):
    abs_threshold = 15
    h[h > abs_threshold] = abs_threshold
    h[h < -abs_threshold] = -abs_threshold
    return h


def cross_entropy_loss(y, tx, w):
    # clipping values to avoid INF loss
    h = sigmoid(clipping(tx @ w))
    return np.squeeze((-y.T @ np.log(h)) + (-(1 - y).T @ np.log(1 - h)))
                  
    
def cross_entropy_gradient(y, tx, w):
    h = sigmoid(tx @ w)
    return (tx.T @ (h - y))


def regularized_cross_entropy_loss(y, tx, w, lambda_):
    return cross_entropy_loss(y, tx, w) + lambda_ * np.squeeze(w.T @ w)


def regularized_cross_entropy_gradient(y, tx, w, lambda_):
    return cross_entropy_gradient(y, tx, w) + 2 * lambda_ * w


def logistic_regression(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """Logistic regression using gradient descent or SGD"""
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gr = cross_entropy_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * gr
            
        loss = cross_entropy_loss(y, tx, w)
            
#        print("SGD ({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
#              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return (w, loss)


def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma, batch_size=1):
    """Regularized logistic regression using gradient descent or SGD"""
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gr = regularized_cross_entropy_gradient(minibatch_y, minibatch_tx, w, lambda_)
            w = w - gamma * gr
            
        loss = regularized_cross_entropy_loss(y, tx, w, lambda_)
            
#        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
#              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return (w, loss)