import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    # Your code here
    n = X.shape[0]
    w = w.reshape(w.shape[0],1)
    y_true = y_true.reshape(y_true.shape[0],1)
    diff = y_true - np.matmul(X,w)
    loss = (1/n)*(np.matmul(np.transpose(diff),diff)) + alpha*np.matmul(np.transpose(w),w)
    return (loss.flatten()).tolist()[0]
    
