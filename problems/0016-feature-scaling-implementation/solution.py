import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here
	means = np.mean(data,axis = 0)
	std = np.std(data,axis = 0)
	mins = np.min(data,axis = 0)
	maxs = np.max(data,axis = 0)
	standardized_data = (data-means)/(std)
	normalized_data = (data-mins)/(maxs-mins)
	
	return standardized_data, normalized_data