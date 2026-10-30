import numpy as np

def to_categorical(x, n_col=None):
	# Your code here
	n_columns = n_col if n_col else np.max(x)+1
	zeros = np.zeros(n_columns)
	result = []
	for i in range(len(x)):
		temp = zeros.copy()
		temp[x[i]] = 1
		result.append(temp)
	return np.array(result)
	