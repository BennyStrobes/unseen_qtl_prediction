import argparse
import numpy as np
import os
import pdb
import tensorflow as tf
import time
import sys






def load_in_tissue_names(gtex_tissue_names_file):
	head_count = 0
	arr = []

	f = open(gtex_tissue_names_file)
	for line in f:
		line = line.rstrip()
		if head_count == 0:
			head_count = head_count + 1
			continue
		arr.append(line)
	f.close()
	return np.asarray(arr)


def load_in_gene_based_model_data(prediction_input_data_summary_filestem, min_snps_per_gene=50):
	arr = []
	for chrom_num in range(1,23):
		f = open(prediction_input_data_summary_filestem + str(chrom_num) + '.txt')
		head_count = 0
		counter = 0
		max_zeds = []
		#indices = []
		for line in f:
			line = line.rstrip()
			data = line.split('\t')
			if head_count == 0:
				head_count = head_count + 1
				continue
			gene_name = data[0]
			snp_summary_file = data[1]
			zed_file = data[2]
			N_eff_file = data[3]
			ld_file = data[4]
			inv_ld_file = data[5]
			borzoi_file = data[6]
			n_snps_per_gene = int(data[8])
			if n_snps_per_gene < min_snps_per_gene:
				continue
			counter = counter + 1

			if np.random.choice(np.arange(8)) == 1:
				arr.append((gene_name, snp_summary_file, zed_file, N_eff_file, ld_file, inv_ld_file, borzoi_file, n_snps_per_gene))
			#indices.append(counter)
		f.close()

	#indices = np.asarray(indices)
	#print(indices)
	#max_zeds = np.asarray(max_zeds)
	print(len(arr))

	return arr


def load_in_gene_based_model_data(prediction_input_data_summary_filestem, min_snps_per_gene=50):
	arr = []
	for chrom_num in range(1,23):
		f = open(prediction_input_data_summary_filestem + str(chrom_num) + '.txt')
		head_count = 0
		counter = 0
		max_zeds = []
		#indices = []
		for line in f:
			line = line.rstrip()
			data = line.split('\t')
			if head_count == 0:
				head_count = head_count + 1
				continue
			gene_name = data[0]
			snp_summary_file = data[1]
			zed_file = data[2]
			N_eff_file = data[3]
			ld_file = data[4]
			inv_ld_file = data[5]
			borzoi_file = data[6]
			n_snps_per_gene = int(data[8])
			if n_snps_per_gene < min_snps_per_gene:
				continue
			counter = counter + 1

			if np.random.choice(np.arange(8)) == 1:
				arr.append((gene_name, snp_summary_file, zed_file, N_eff_file, ld_file, inv_ld_file, borzoi_file, n_snps_per_gene))
			#indices.append(counter)
		f.close()

	#indices = np.asarray(indices)
	#print(indices)
	#max_zeds = np.asarray(max_zeds)
	print(len(arr))

	return arr

def split_train_and_val_gene_based_model_data(train_val_gene_based_model_data, use_held_out_genes_for_validation):
	if use_held_out_genes_for_validation:
		tot_genes = len(train_val_gene_based_model_data)
		n_train_genes = int(np.floor(tot_genes*.8))
		train_gene_based_model_data = train_val_gene_based_model_data[:n_train_genes]
		val_gene_based_model_data = train_val_gene_based_model_data[n_train_genes:]
	else:
		train_gene_based_model_data = train_val_gene_based_model_data.copy()
		val_gene_based_model_data = train_val_gene_based_model_data.copy()
	return train_gene_based_model_data, val_gene_based_model_data


def evaluate_model(gene_based_model_data, train_gene_based_model_data, val_gene_based_model_data, test_gene_based_model_data, test_tissue_index, borzoi_eval_output_stem, borzoi_target_index):
	train_gene_names = {gene_data[0] for gene_data in train_gene_based_model_data}
	val_gene_names = {gene_data[0] for gene_data in val_gene_based_model_data}
	test_gene_names = {gene_data[0] for gene_data in test_gene_based_model_data}


	output_file = borzoi_eval_output_stem + '_all_gene_test_tissue_evaluation.txt'
	variant_output_file = borzoi_eval_output_stem + '_all_variant_gene_pairs_test_tissue_evaluation.txt'

	with open(output_file, 'w') as t, open(variant_output_file, 'w') as v:
		t.write('gene_name\tgene_split\tn_snps\tloss\tcorr\tpred_expr_corr\n')
		v.write('gene_name\tvariant_name\tobs_gene_zed\tpred_gene_zed\tgene_split\n')
		for gene_name, gene_snp_summary_file, gene_zed_file, gene_N_eff_file, gene_LD_file, gene_inv_LD_file, gene_borzoi_pred_file, n_gene_snps in gene_based_model_data:
			if gene_name in train_gene_names:
				gene_split = 'train'
			elif gene_name in val_gene_names:
				gene_split = 'validation'
			elif gene_name in test_gene_names:
				gene_split = 'test'
			else:
				gene_split = 'unknown'

			gene_LD = np.load(gene_LD_file)
			gene_inv_LD = np.load(gene_inv_LD_file)
			borzoi_mat = np.load(gene_borzoi_pred_file)
			borzoi_mat = borzoi_mat[:, borzoi_target_index:(borzoi_target_index+1)]
			gene_zeds = np.load(gene_zed_file)
			gene_N_eff = np.load(gene_N_eff_file)
			gene_snp_summary = np.loadtxt(gene_snp_summary_file, dtype=str)
			gene_variant_names = gene_snp_summary[1:, 0]
			gene_afs = gene_snp_summary[1:, -1].astype(float)
			valid_row_indices = np.where(~np.isnan(borzoi_mat[:,0]))[0]
			if len(valid_row_indices) == 0:
				t.write(gene_name + '\t' + gene_split + '\t' + str(n_gene_snps) + '\tnan\tnan\tnan\n')
				continue

			gene_LD = gene_LD[valid_row_indices, :][:, valid_row_indices]
			gene_borzoi_preds = borzoi_mat[valid_row_indices, :]
			gene_N_eff = gene_N_eff[valid_row_indices, :][:, [test_tissue_index]]
			gene_zeds = gene_zeds[valid_row_indices, :][:, [test_tissue_index]]
			gene_variant_names = gene_variant_names[valid_row_indices]
			gene_afs = gene_afs[valid_row_indices]

			gene_LD_tf = tf.convert_to_tensor(gene_LD.astype(np.float32))
			gene_inv_LD_tf = tf.convert_to_tensor(gene_inv_LD.astype(np.float32))
			gene_borzoi_preds_tf = tf.convert_to_tensor(gene_borzoi_preds.astype(np.float32))
			gene_N_eff_tf = tf.convert_to_tensor(gene_N_eff.astype(np.float32))
			gene_zeds_tf = tf.convert_to_tensor(gene_zeds.astype(np.float32))
			gene_afs_tf = tf.convert_to_tensor(gene_afs.astype(np.float32))

			genotype_sd = tf.sqrt(2.0 * gene_afs_tf * (1.0 - gene_afs_tf))
			beta_std_mat = gene_borzoi_preds_tf * genotype_sd[:, None]
			full_pred_gene_zeds_mat = tf.sqrt(gene_N_eff_tf) * tf.matmul(gene_LD_tf, beta_std_mat)

			valid_tissues = ~tf.reduce_any(tf.math.is_nan(gene_zeds_tf), axis=0)
			obs_gene_zeds_mat = tf.boolean_mask(gene_zeds_tf, valid_tissues, axis=1)
			pred_gene_zeds_mat = tf.boolean_mask(full_pred_gene_zeds_mat, valid_tissues, axis=1)
			residuals = obs_gene_zeds_mat - pred_gene_zeds_mat
			gene_loss = tf.reduce_sum(residuals * tf.matmul(gene_inv_LD_tf, residuals)).numpy()

			obs_gene_zeds = tf.reshape(obs_gene_zeds_mat, [-1])
			pred_gene_zeds = tf.reshape(pred_gene_zeds_mat, [-1])
			causal_beta = beta_std_mat[:,0].numpy()

			if tf.size(obs_gene_zeds) > 1:
				obs_centered = obs_gene_zeds - tf.reduce_mean(obs_gene_zeds)
				pred_centered = pred_gene_zeds - tf.reduce_mean(pred_gene_zeds)
				eps = tf.constant(1e-8, dtype=obs_gene_zeds.dtype)
				gene_corr = (
					tf.reduce_sum(obs_centered * pred_centered) /
					(tf.sqrt(tf.reduce_sum(tf.square(obs_centered)) + eps) * tf.sqrt(tf.reduce_sum(tf.square(pred_centered)) + eps))
				).numpy()
				std_beta = obs_gene_zeds.numpy()/(np.sqrt(tf.reshape(tf.boolean_mask(gene_N_eff_tf, valid_tissues, axis=1), [-1]).numpy()))
				expr_corr = np.dot(std_beta, causal_beta)/np.sqrt(np.dot(np.dot(causal_beta, gene_LD), causal_beta))
			else:
				gene_corr = np.nan
				expr_corr = np.nan

			t.write(gene_name + '\t' + gene_split + '\t' + str(n_gene_snps) + '\t' + str(gene_loss) + '\t' + str(gene_corr) + '\t' + str(expr_corr) + '\n')
			full_obs_gene_zeds = gene_zeds_tf.numpy()[:, 0]
			full_pred_gene_zeds = full_pred_gene_zeds_mat.numpy()[:, 0]
			for variant_name, obs_gene_zed, pred_gene_zed in zip(gene_variant_names, full_obs_gene_zeds, full_pred_gene_zeds):
				v.write(gene_name + '\t' + variant_name + '\t' + str(obs_gene_zed) + '\t' + str(pred_gene_zed) + '\t' + gene_split + '\n')




###########################
# Command line args
#############################
test_tissue = sys.argv[1]
borzoi_target_index = int(sys.argv[2])
borzoi_eval_output_stem = sys.argv[3]
prediction_input_data_summary_filestem = sys.argv[4]
gtex_tissue_names_file = sys.argv[5]


np.random.seed(1)




# Load in all tissues names
all_tissue_names = load_in_tissue_names(gtex_tissue_names_file)

# Get index of test tissue
test_tissue_indices = np.where(all_tissue_names == test_tissue)[0]
if len(test_tissue_indices) != 1:
	print('assumption eroror')
	pdb.set_trace()
test_tissue_index = test_tissue_indices[0]



##############
# Can delete in future (currently done to get same random seed)
###############*#*#*#*#
n_val_tissues=5
# Get indices of training + val tissue
train_val_tissue_indices = np.arange(len(all_tissue_names))
train_val_tissue_indices = np.delete(train_val_tissue_indices, test_tissue_index)
val_tissue_indices = np.sort(np.random.choice(train_val_tissue_indices, size=n_val_tissues, replace=False))
train_tissue_indices = train_val_tissue_indices[np.isin(train_val_tissue_indices, val_tissue_indices, invert=True)]
train_tissue_names = all_tissue_names[train_tissue_indices]
val_tissue_names = all_tissue_names[val_tissue_indices]
##############
# END delete in future
###############*#*#*#*#



# Load in gene-based model training/evaluation data
gene_based_model_data = load_in_gene_based_model_data(prediction_input_data_summary_filestem, min_snps_per_gene=50)


tot_n_genes = len(gene_based_model_data)
n_train_val_genes = int(np.floor(tot_n_genes*.8))
# Split into train/val and test
train_val_gene_based_model_data = gene_based_model_data[:n_train_val_genes]
test_gene_based_model_data = gene_based_model_data[n_train_val_genes:]

use_held_out_genes_for_validation=True
train_gene_based_model_data, val_gene_based_model_data = split_train_and_val_gene_based_model_data(train_val_gene_based_model_data, use_held_out_genes_for_validation)



evaluate_model(gene_based_model_data, train_gene_based_model_data, val_gene_based_model_data, test_gene_based_model_data, test_tissue_index, borzoi_eval_output_stem, borzoi_target_index)

