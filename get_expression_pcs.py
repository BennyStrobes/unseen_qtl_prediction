import numpy as np
import os
import sys
import pdb
from sklearn.decomposition import PCA




def extract_tissue_names(training_tissue_file):
	f = open(training_tissue_file)
	head_count = 0
	arr = []
	for line in f:
		line = line.rstrip()
		if head_count == 0:
			head_count = head_count + 1
			continue
		arr.append(line)
	f.close()
	return np.asarray(arr)


def filter_expression_to_set_of_tissues(tissues, processed_all_sample_expression_file, output_expression_file):
	# Get dictionary with tissue names as keys
	tissue_dicti = {}
	for tissue in tissues:
		if tissue in tissue_dicti:
			print('assumption eroror')
			pdb.set_trace()
		tissue_dicti[tissue] = 1


	# Open input and output files
	f = open(processed_all_sample_expression_file)
	t = open(output_expression_file,'w')
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			all_samps = np.asarray(data[2:])
			valid_samps = []
			for samp_name in all_samps:
				samp_tissue = samp_name.split(':')[1]
				if samp_tissue in tissue_dicti:
					valid_samps.append(True)
				else:
					valid_samps.append(False)
			valid_samps = np.asarray(valid_samps)
			t.write(data[0] + '\t' + data[1] + '\t' + '\t'.join(all_samps[valid_samps]) + '\n')
			continue
		all_samps = np.asarray(data[2:])
		t.write(data[0] + '\t' + data[1] + '\t' + '\t'.join(all_samps[valid_samps]) + '\n')

	f.close()
	t.close()

	return


def extract_gene_expression_data(expression_file):
	f = open(expression_file)
	head_count = 0
	expr = []
	samples = []
	genes = []
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			samples_names = np.asarray(data[2:])
			continue
		genes.append(data[0])
		expr.append(np.asarray(data[2:]).astype(float))
	f.close()

	genes = np.asarray(genes)
	expr  = np.asarray(expr)

	return expr, samples_names, genes

def pca_train(E_train, n_components):
	"""
	Fit PCA on training gene expression data and return:
	  - low-dim representation of training samples
	  - fitted PCA object
	  - per-gene means and stds used for standardization

	Parameters
	----------
	E_train : np.ndarray, shape (N_train, n_genes)
		Training expression matrix (log-scale, not standardized).
	n_components : int
		Number of principal components to keep.

	Returns
	-------
	Z_train : np.ndarray, shape (N_train, n_components)
		PCA scores for training samples.
	pca : sklearn.decomposition.PCA
		Fitted PCA model.
	gene_means : np.ndarray, shape (n_genes,)
		Mean expression per gene (computed from E_train).
	gene_stds : np.ndarray, shape (n_genes,)
		Std dev per gene (computed from E_train, with zeros handled).
	"""
	# 1. Compute train-only mean and std per gene
	gene_means = E_train.mean(axis=0)
	gene_stds = E_train.std(axis=0, ddof=0)

	# Handle zero-variance genes to avoid division by zero
	gene_stds_safe = gene_stds.copy()
	gene_stds_safe[gene_stds_safe == 0] = 1.0

	# 2. Z-score standardize train data
	E_train_z = (E_train - gene_means) / gene_stds_safe

	# 3. Fit PCA
	pca = PCA(n_components=n_components)
	pca.fit(E_train_z)

	# 4. Transform train data into PCA space
	Z_train = pca.transform(E_train_z)

	return Z_train, pca, gene_means, gene_stds_safe


def pca_test(E_test, pca, gene_means, gene_stds):
	"""
	Project test gene expression data into an existing PCA space,
	using train-derived gene means and stds.

	Parameters
	----------
	E_test : np.ndarray, shape (N_test, n_genes)
		Test expression matrix (same genes/columns as E_train).
	pca : sklearn.decomposition.PCA
		PCA model previously fitted on standardized E_train.
	gene_means : np.ndarray, shape (n_genes,)
		Train-derived per-gene means.
	gene_stds : np.ndarray, shape (n_genes,)
		Train-derived per-gene stds (with zeros already handled).

	Returns
	-------
	Z_test : np.ndarray, shape (N_test, n_components)
		PCA scores for test samples in the same space as train.
	"""
	# 1. Standardize test using *train* stats
	E_test_z = (E_test - gene_means) / gene_stds

	# 2. Project into PCA space
	Z_test = pca.transform(E_test_z)

	return Z_test

def print_pcs_to_output(Z_mat, sample_names, expression_pc_file):
	# Quick error checking
	if Z_mat.shape[0] != len(sample_names):
		print('assumptino erooror')
		pdb.set_trace()

	t = open(expression_pc_file,'w')
	t.write('PC_name\t' + '\t'.join(sample_names) + '\n')

	n_pcs = Z_mat.shape[1]
	for pc_iter in range(n_pcs):
		pc_val = Z_mat[:, pc_iter]
		t.write('PC' + str(pc_iter) + '\t' + '\t'.join(pc_val.astype(str)) + '\n')
	t.close()
	return



#######################
# Command line args
#######################
processed_all_sample_expression_file = sys.argv[1]
expression_pc_file = sys.argv[2]
n_pcs = int(sys.argv[3])



# Extract gene expression data
E_train, samples_train, genes_train = extract_gene_expression_data(processed_all_sample_expression_file)


# Run PCA
# Train phase
Z_train, pca, gene_means, gene_stds = pca_train(np.transpose(E_train), n_components=n_pcs)


# Print PCs to output
print_pcs_to_output(Z_train, samples_train, expression_pc_file)






