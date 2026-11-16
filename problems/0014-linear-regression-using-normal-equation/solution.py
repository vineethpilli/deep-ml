import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    # Your code here, make sure to round
    #theta = (xT.x)^-1.xT.y
    X = np.array(X)
    y = np.array(y)
    xtransposex = np.matmul(np.transpose(X),X)
    theta = np.matmul(np.matmul(np.linalg.inv(xtransposex),np.transpose(X)),y)
    return theta