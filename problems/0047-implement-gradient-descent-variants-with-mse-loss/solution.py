import numpy as np
def one_gradient_step(X,y,weights,learning_rate,start_ind,end_ind):
    if end_ind>X.shape[0]:
        end_ind = X.shape[0]
    X_dash = X[start_ind:end_ind]
    y_dash = y[start_ind:end_ind]
    k = end_ind-start_ind
    gradient = (2*learning_rate/k)*(np.matmul(np.matmul(np.transpose(X_dash),X_dash),weights)-np.matmul(np.transpose(X_dash),y_dash))
    weights = weights - gradient
    return weights

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    # Your code here
    y = y.reshape(y.shape[0],1)
    m = X.shape[0]
    weights = weights.reshape(weights.shape[0],1)
    if method == 'batch':
        for i in range(n_epochs):
            weights = one_gradient_step(X,y,weights,learning_rate,0,m)
        return (weights.flatten()).tolist()
    elif method == 'stochastic':
        for i in range(n_epochs):
            for k in range(m):
                weights = one_gradient_step(X,y,weights,learning_rate,k,k+1)
        return (weights.flatten()).tolist()
    else:
        for i in range(n_epochs):
            for k in range(0,m,batch_size):
                weights = one_gradient_step(X,y,weights,learning_rate,k,k+batch_size)
        return (weights.flatten()).tolist()
    pass
