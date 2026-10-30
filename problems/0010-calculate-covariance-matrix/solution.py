import numpy as np
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
		N = len(vectors[0])
		# print(N)
		averages = []
		for i in range(len(vectors)):
			averages.append(np.mean(vectors[i]))
		feature_means = np.array(averages)
		feature_means = feature_means.reshape(len(averages),1)
		matrix = np.array(vectors)
		
		matrix = matrix-feature_means
		ans = np.matmul(matrix,matrix.T)
		ans = ans/(N-1)
		return ans