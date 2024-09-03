import scipy.io as sio
import numpy as np

# Load the original .mat file
data = sio.loadmat('recipe2-emd_tr_te_split.mat')

# Sampling the attributes corresponding to X, Y, BOW_X, and words
X = data['X']
Y = data['Y']
BOW_X = data['BOW_X']
the_words = data['the_words']

# Define the reduction fraction
frac = 0.1
num_elements = len(X[0])
num_sampled_elements = int(num_elements * frac)
sampled_indices = np.random.choice(num_elements, num_sampled_elements, replace=False)

# regenerate the indices of the 5-fold sampled elements in the training set (TR) and test set (TE)
TR = data['TR']
TR_size = TR.shape[1]
TR_size_reduced = int(TR_size * frac)
TE = data['TE']
TE_size = TE.shape[1]
TE_size_reduced = int(TE_size * frac)

sampled_TR_indices = []
sampled_TE_indices = []
num_folds = TR.shape[0]

np.random.seed(0)

for i in range(num_folds):
    indices = np.arange(TR_size_reduced + TE_size_reduced) + 1
    np.random.shuffle(indices)
    sampled_TR_indices.append(indices[:TR_size_reduced])
    sampled_TE_indices.append(indices[TR_size_reduced:])
TR = np.array(sampled_TR_indices)
TE = np.array(sampled_TE_indices)

X = X[:, sampled_indices]       # Keep the original name 'X'
Y = Y[:, sampled_indices]       # Keep the original name 'Y'
BOW_X = BOW_X[:, sampled_indices]  # Keep the original name 'BOW_X'
the_words = the_words[:, sampled_indices]  # Keep the original name 'the_words'

# Save the sampled dataset with the original names
sio.savemat('recipe2-emd_tr_te_split_reduced.mat', {'TR': TR, 'TE': TE, 'X': X, 'Y': Y, 'BOW_X': BOW_X, 'the_words': the_words})
