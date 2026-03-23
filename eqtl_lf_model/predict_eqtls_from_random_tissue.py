import numpy as np
import os
import sys
import pdb
import argparse
import tensorflow as tf
from tensorflow.keras import layers

def load_in_eqtl_data(file_name):
	f = open(file_name)
	head_count = 0
	effect_mat = []
	variant_names = []
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			tissue_names = data[1:]
			continue
		variant_names.append(data[0])
		effect_mat.append(np.asarray(data[1:]).astype(float))
	f.close()

	effect_mat = np.asarray(effect_mat)
	variant_names = np.asarray(variant_names)
	tissue_names = np.asarray(tissue_names)

	return effect_mat, variant_names, tissue_names



def genomic_block_bootstrap_avg(arr, n_blocks=100, n_boots=2000):
	blocks = np.array_split(arr, n_blocks)
	n_blocks = len(blocks)

	bs_means = []
	for bs_iter in range(n_boots):
		bs_blocks = np.random.choice(np.arange(n_blocks), size=n_blocks, replace=True)
		boot_arr = np.concatenate([blocks[i] for i in bs_blocks])
		bs_means.append(np.mean(boot_arr))
	bs_means = np.asarray(bs_means)

	return np.mean(bs_means), np.std(bs_means)

def genomic_block_bootstrap_corr(pred, truth, n_blocks=100, n_boots=2000):
	arr = np.transpose(np.vstack((pred, truth)))
	blocks = np.array_split(arr, n_blocks,axis=0)

	bs_corrz = []
	for bs_iter in range(n_boots):
		bs_blocks = np.random.choice(np.arange(n_blocks), size=n_blocks, replace=True)	
		boot_arr = np.concatenate([blocks[i] for i in bs_blocks])
		bs_corrz.append(np.corrcoef(boot_arr[:,0], boot_arr[:,1])[0,1])
	bs_corrz = np.asarray(bs_corrz)
	return np.mean(bs_corrz), np.std(bs_corrz)

def get_test_losses(model_preds, model_preds_mask, beta_test, se_test, mask_test):

	# Compute test loss
	test_losses = []
	test_losses_ses = []
	test_corrz = []
	test_corrz_ses = []
	for col_iter in range(beta_test.shape[1]):
		# Subset to just this column
		tmp_model_preds = model_preds[:, col_iter]
		tmp_mask_preds = model_preds_mask[:, col_iter]
		tmp_beta_test = beta_test[:, col_iter]
		tmp_se_test = se_test[:, col_iter]
		tmp_mask_test = mask_test[:, col_iter]
		tmp_mask = (tmp_mask_preds) & (tmp_mask_test)

		############
		# Get loss
		squared_resids = np.square(tmp_model_preds[tmp_mask] - tmp_beta_test[tmp_mask])
		denom = np.square(tmp_se_test[tmp_mask])
		per_snp_loss = squared_resids/denom
		avg_loss = np.mean(per_snp_loss)
		avg_loss_bs, avg_loss_bs_se = genomic_block_bootstrap_avg(per_snp_loss, n_blocks=100, n_boots=1000)
		# Append to array
		test_losses.append(avg_loss)
		test_losses_ses.append(avg_loss_bs_se)


		############
		# Get correlation
		avg_corr = np.corrcoef(tmp_model_preds[tmp_mask], tmp_beta_test[tmp_mask])[0,1]
		avg_corr_bs, avg_corr_bs_se = genomic_block_bootstrap_corr(tmp_model_preds[tmp_mask], tmp_beta_test[tmp_mask], n_blocks=100, n_boots=1000)
		test_corrz.append(avg_corr)
		test_corrz_ses.append(avg_corr_bs_se)


	test_losses = np.asarray(test_losses)
	test_losses_ses = np.asarray(test_losses_ses)
	test_corrz = np.asarray(test_corrz)
	test_corrz_ses = np.asarray(test_corrz_ses)



	return test_losses, test_losses_ses, test_corrz, test_corrz_ses, tmp_model_preds[tmp_mask], tmp_beta_test[tmp_mask], tmp_se_test[tmp_mask]

def load_in_ordered_tissue_names(tissue_file, expression_file):
	tissue_names = []
	f = open(tissue_file)
	head_count = 0
	for line in f:
		line = line.rstrip()
		if head_count == 0:
			head_count = head_count + 1
			continue
		tissue_names.append(line)
	f.close()
	tissue_names = np.asarray(tissue_names)

	f = open(expression_file)
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		alt_tissues = []
		for ele in data[1:]:
			alt_tissues.append(ele.split(':')[1])
		alt_tissues = np.asarray(alt_tissues)
		if np.array_equal(alt_tissues, tissue_names) == False:
			print('assumption erororo')
			pdb.set_trace()
		break
	f.close()

	return tissue_names

def get_training_validation_and_test_tissue_indices(tissue_names, test_tissues, n_validation_tissues):
	test_idx = []
	non_test_idx = []

	for ii, tissue_name in enumerate(tissue_names):
		if tissue_name in test_tissues:
			test_idx.append(ii)
		else:
			non_test_idx.append(ii)
	test_idx = np.asarray(test_idx)
	non_test_idx = np.asarray(non_test_idx)

	val_tissue_idx = np.sort(np.random.choice(non_test_idx,size=n_validation_tissues, replace=False))

	train_tissue_idx = []
	for tissue_idx in non_test_idx:
		if tissue_idx not in val_tissue_idx:
			train_tissue_idx.append(tissue_idx)
	train_tissue_idx = np.asarray(train_tissue_idx)

	return train_tissue_idx, val_tissue_idx, test_idx


def get_nearest_training_tissue_index(X_vec, X_train_t):
	distances = []
	for tiss_iter in range(X_train_t.shape[0]):
		distance = np.sum(np.square(X_vec - X_train_t[tiss_iter, :]))
		distances.append(distance)
	distances = np.asarray(distances)
	return np.argmin(distances), np.min(distances)



################################################################################
# main
################################################################################
def main():
	######################
	# Command line args
	######################
	# Necessary
	parser = argparse.ArgumentParser()
	parser.add_argument('--eqtl_effect_size_file', default='', type=str,
						help='Matrix of estimated eqtl effect sizes')
	parser.add_argument('--eqtl_se_file', default='', type=str,
						help='Matrix of estimated eqtl effect size standard errors')
	parser.add_argument('--expression_file', default='', type=str,
						help='Expression file')
	parser.add_argument('--output_stem', default='', type=str,
						help='Path to output file stem')
	parser.add_argument('--tissue_file', default='', type=str,
						help='File containing ordered tissue names')
	parser.add_argument('--test_tissue_list', default='None', type=str,
						help=';-sepearted list of test tissues')
	parser.add_argument('--random_seed', default=1, type=int,
						help='Number of hidden layers in tissue MLP')
	args = parser.parse_args()


	np.random.seed(args.random_seed)

	# Load in tissue names
	tissue_names = load_in_ordered_tissue_names(args.tissue_file, args.expression_file)

	# Load in estimated eqtl effects
	eqtl_beta_hat, variant_names_beta, tissue_names_beta = load_in_eqtl_data(args.eqtl_effect_size_file)
	eqtl_beta_hat_se, variant_names_se, tissue_names_se = load_in_eqtl_data(args.eqtl_se_file)

	# Load in expression data
	expression_mat = (np.loadtxt(args.expression_file, dtype=str,delimiter='\t')[2:,:][:,1:]).astype(float)

	# Now split into training, validation, and test data
	orig_test_tissues = np.asarray(args.test_tissue_list.split(';'))
	train_tissue_idx, val_tissue_idx, test_tissue_idx = get_training_validation_and_test_tissue_indices(tissue_names, orig_test_tissues, 0)	
	train_tissues = tissue_names[train_tissue_idx]
	val_tissues = tissue_names[val_tissue_idx]
	test_tissues = tissue_names[test_tissue_idx]


	# Slice eQTL data into TRAIN, VAL, and TEST sets
	# Limit to variant-gene pairs we have at least 1 tissue for
	tmp_beta_train = eqtl_beta_hat[:, train_tissue_idx]   # (N, T_train)
	valid_variant_gene_pairs = np.sum(np.isnan(tmp_beta_train) == False,axis=1) > 0

	beta_train = eqtl_beta_hat[:, train_tissue_idx][valid_variant_gene_pairs, :]   # (N, T_train)
	se_train   = eqtl_beta_hat_se[:,   train_tissue_idx][valid_variant_gene_pairs, :]   # (N, T_train)
	X_train_t  = np.transpose(expression_mat[:, train_tissue_idx])    # (T_train, G)
	beta_test   = eqtl_beta_hat[:, test_tissue_idx][valid_variant_gene_pairs, :]    # (N, T_test)
	se_test     = eqtl_beta_hat_se[:,   test_tissue_idx][valid_variant_gene_pairs, :]     # (N, T_test)
	X_test_t    = np.transpose(expression_mat[:, test_tissue_idx])      # (T_test, G)



	# Build masks of observed cells (True where we have QTL data)
	mask_train = ~np.isnan(beta_train)            # (N, T_train)
	mask_test  = ~np.isnan(beta_test)              # (N, T_test)

	# Replace NaNs (they'll be masked out anyway)
	beta_train = np.nan_to_num(beta_train, nan=0.0).astype(np.float32)
	se_train   = np.nan_to_num(se_train,   nan=1.0).astype(np.float32)
	beta_test  = np.nan_to_num(beta_test,   nan=0.0).astype(np.float32)
	se_test    = np.nan_to_num(se_test,     nan=1.0).astype(np.float32)

	mask_train = mask_train.astype(np.bool_)
	mask_test   = mask_test.astype(np.bool_)
	X_train_t  = X_train_t.astype(np.float32)
	X_test_t    = X_test_t.astype(np.float32)

	# "Training"
	preds = []
	preds_mask = []
	nearest_tissues = []
	nearest_tissue_distances = []
	for test_iter, tissue_name in enumerate(test_tissues):
		nearest_training_tissue_index, min_dist = get_nearest_training_tissue_index(X_test_t[test_iter,:], X_train_t)
		nearest_training_tissue_index = np.random.choice(X_train_t.shape[0])
		preds.append(np.copy(beta_train[:, nearest_training_tissue_index]))
		preds_mask.append(np.copy(mask_train[:, nearest_training_tissue_index]))
		nearest_tissues.append(train_tissues[nearest_training_tissue_index])
		nearest_tissue_distances.append(min_dist)

	preds = np.transpose(np.asarray(preds))
	preds_mask = np.transpose(np.asarray(preds_mask))
	nearest_tissues = np.asarray(nearest_tissues)
	nearest_tissue_distances = np.asarray(nearest_tissue_distances)


	# Get test losses
	test_losses, test_losses_ses, test_corrz, test_corrz_ses, test_beta_preds, test_betas, test_beta_se = get_test_losses(preds, preds_mask, beta_test, se_test, mask_test)

	# Print test losses to output
	# Open output file
	test_loss_output_file = args.output_stem + '_random_tissue_pred_test_loss_summary.txt'
	t = open(test_loss_output_file,'w')
	t.write('tissue\tloss\tloss_se\tcorrelation\tcorrelation_se\n')
	# Loop through tissues
	for ii, tissue_name in enumerate(test_tissues):
		t.write(tissue_name + '\t' + str(test_losses[ii]) + '\t' + str(test_losses_ses[ii]) + '\t' + str(test_corrz[ii]) + '\t' + str(test_corrz_ses[ii]) + '\n')
	t.close()
	print(test_loss_output_file)



	# Print test losses to output
	# Open output file
	test_loss_output_file = args.output_stem + '_random_tissue_test_preds.txt'
	t = open(test_loss_output_file,'w')
	t.write('beta\tbeta_se\tpred_beta\n')
	# Loop through tissues
	for ii, test_beta_pred in enumerate(test_beta_preds):
		t.write(str(test_betas[ii]) + '\t' + str(test_beta_se[ii]) + '\t' + str(test_beta_pred) + '\n')
	t.close()


	return



################################################################################
# __main__
################################################################################
if __name__ == '__main__':
	main()
