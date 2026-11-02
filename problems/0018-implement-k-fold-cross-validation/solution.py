import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Your code here
    fold_size = int(n_samples/k)
    first_fold_size = int(fold_size+(n_samples%k))
    
    result_list = []
    
    data_samples = [i for i in range(n_samples)]
    
    if shuffle:
        data_samples = np.array(data_samples)
        np.random.shuffle(data_samples)
        data_samples = data_samples.tolist()
    

    result_list.append((data_samples[first_fold_size:],data_samples[0:first_fold_size]))
    
    j = first_fold_size
    
    for n_folds in range(1,k):
        try:
            fold_list = data_samples[j:j+fold_size]
            non_fold_list = data_samples[:j]+data_samples[j+fold_size:]
            j += fold_size
            result_list.append((non_fold_list,fold_list))
        except Exception as e:
            print(e)
    
    return result_list
