import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def evaluate_model(gene_based_model_data, test_tissue_index, borzoi_eval_output_stem, borzoi_target_index):

	output_file = borzoi_eval_output_stem + '_expression_correlation.txt'

	with open(output_file, 'w') as t:
		t.write('gene_name\tn_snps\tmax_abs_borzoi\tpred_expr_corr\n')
		for gene_name, gene_snp_summary_file, gene_zed_file, gene_N_eff_file, gene_LD_file, gene_inv_LD_file, gene_borzoi_pred_file, n_gene_snps in gene_based_model_data:

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
				t.write(gene_name + '\t' + str(n_gene_snps) + '\tnan\tnan\n')
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
			#full_pred_gene_zeds_mat = tf.sqrt(gene_N_eff_tf) * tf.matmul(gene_LD_tf, beta_std_mat)

			valid_tissues = ~tf.reduce_any(tf.math.is_nan(gene_zeds_tf), axis=0)
			obs_gene_zeds_mat = tf.boolean_mask(gene_zeds_tf, valid_tissues, axis=1)
			#pred_gene_zeds_mat = tf.boolean_mask(full_pred_gene_zeds_mat, valid_tissues, axis=1)

			obs_gene_zeds = tf.reshape(obs_gene_zeds_mat, [-1])
			#pred_gene_zeds = tf.reshape(pred_gene_zeds_mat, [-1])
			causal_beta = beta_std_mat[:,0].numpy()

			if tf.size(obs_gene_zeds) > 1:
				std_beta = obs_gene_zeds.numpy()/(np.sqrt(tf.reshape(tf.boolean_mask(gene_N_eff_tf, valid_tissues, axis=1), [-1]).numpy()))
				
				expr_corr = np.dot(std_beta, causal_beta)/np.sqrt(np.dot(np.dot(causal_beta, gene_LD), causal_beta))

			else:
				gene_corr = np.nan
				expr_corr = np.nan

			t.write(gene_name  + '\t' + str(n_gene_snps) + '\t' + str(np.max(np.abs(gene_borzoi_preds)))  + '\t' + str(expr_corr) + '\n')
	
	print(output_file)

	return


def make_expression_correlation_barplot(expression_correlation_file, bin_edges=None):
	data = np.genfromtxt(expression_correlation_file, dtype=str, delimiter='\t', skip_header=1)
	if data.size == 0:
		print('No rows in ' + expression_correlation_file + '; skipping plot')
		return
	if data.ndim == 1:
		data = data.reshape(1, -1)

	max_abs_borzoi = data[:, 2].astype(float)
	expr_corr = data[:, 3].astype(float)

	valid = (~np.isnan(max_abs_borzoi)) & (~np.isnan(expr_corr))
	max_abs_borzoi = max_abs_borzoi[valid]
	expr_corr = expr_corr[valid]

	if len(max_abs_borzoi) == 0:
		print('No valid rows in ' + expression_correlation_file + '; skipping plot')
		return

	if bin_edges is None:
		bin_edges = np.asarray([0.0, 0.01, 0.05, 0.1, .15, 0.25, 4.0])
	else:
		bin_edges = np.asarray(bin_edges)
	n_bins = len(bin_edges) - 1

	bin_indices = np.digitize(max_abs_borzoi, bin_edges[1:-1], right=False)

	bin_means = []
	bin_cis = []
	bin_labels = []
	for bin_num in range(n_bins):
		bin_mask = bin_indices == bin_num
		bin_values = expr_corr[bin_mask]
		n_bin_genes = len(bin_values)
		if len(bin_values) == 0:
			bin_means.append(np.nan)
			bin_cis.append(0.0)
		else:
			bin_means.append(np.mean(bin_values))
			if len(bin_values) == 1:
				sem = 0.0
			else:
				sem = np.std(bin_values, ddof=1)/np.sqrt(len(bin_values))
			bin_cis.append(1.96*sem)
		bin_label = '[' + '{0:.3g}'.format(bin_edges[bin_num]) + ', ' + '{0:.3g}'.format(bin_edges[bin_num + 1]) + (')' if bin_num < (n_bins - 1) else ']')
		bin_labels.append(bin_label + '\nN=' + str(n_bin_genes))

	bin_means = np.asarray(bin_means)
	bin_cis = np.asarray(bin_cis)

	fig, ax = plt.subplots(figsize=(12, 6))
	x = np.arange(n_bins)
	ax.bar(x, bin_means, yerr=bin_cis, capsize=4, color='#4C78A8', edgecolor='black')
	ax.set_xticks(x)
	ax.set_xticklabels(bin_labels, rotation=45, ha='right')
	ax.set_xlabel('max_abs_borzoi bin')
	ax.set_ylabel('Average expression correlation')
	ax.set_title('Average expression correlation by max_abs_borzoi')
	ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1)
	plt.tight_layout()

	output_file = expression_correlation_file.replace('_expression_correlation.txt', '_expression_correlation_by_max_abs_borzoi_bin_barplot.pdf')
	plt.savefig(output_file)
	plt.close(fig)

	print(output_file)

	return


def make_expression_correlation_violinplot(expression_correlation_file, bin_edges=None):
	data = np.genfromtxt(expression_correlation_file, dtype=str, delimiter='\t', skip_header=1)
	if data.size == 0:
		print('No rows in ' + expression_correlation_file + '; skipping plot')
		return
	if data.ndim == 1:
		data = data.reshape(1, -1)

	max_abs_borzoi = data[:, 2].astype(float)
	expr_corr = data[:, 3].astype(float)

	valid = (~np.isnan(max_abs_borzoi)) & (~np.isnan(expr_corr))
	max_abs_borzoi = max_abs_borzoi[valid]
	expr_corr = expr_corr[valid]


	if len(max_abs_borzoi) == 0:
		print('No valid rows in ' + expression_correlation_file + '; skipping plot')
		return

	if bin_edges is None:
		bin_edges = np.asarray([0.0, 0.01, 0.05, 0.1, .15, 0.25, 4.0])
	else:
		bin_edges = np.asarray(bin_edges)
	n_bins = len(bin_edges) - 1

	bin_indices = np.digitize(max_abs_borzoi, bin_edges[1:-1], right=False)

	bin_values = []
	bin_labels = []
	positions = []
	for bin_num in range(n_bins):
		bin_mask = bin_indices == bin_num
		curr_bin_values = expr_corr[bin_mask]
		if len(curr_bin_values) > 0:
			bin_values.append(curr_bin_values)
			positions.append(bin_num)
		bin_label = '[' + '{0:.3g}'.format(bin_edges[bin_num]) + ', ' + '{0:.3g}'.format(bin_edges[bin_num + 1]) + (')' if bin_num < (n_bins - 1) else ']')
		bin_labels.append(bin_label + '\nN=' + str(len(curr_bin_values)))

	fig, ax = plt.subplots(figsize=(12, 6))
	if len(bin_values) > 0:
		parts = ax.violinplot(bin_values, positions=positions, widths=0.8, showmeans=False, showmedians=True, showextrema=False)
		for pc in parts['bodies']:
			pc.set_facecolor('#4C78A8')
			pc.set_edgecolor('black')
			pc.set_alpha(0.75)

	ax.set_xticks(np.arange(n_bins))
	ax.set_xticklabels(bin_labels, rotation=45, ha='right')
	ax.set_xlabel('max_abs_borzoi bin')
	ax.set_ylabel('Expression correlation')
	ax.set_title('Expression correlation distribution by max_abs_borzoi')
	ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1)
	plt.tight_layout()

	output_file = expression_correlation_file.replace('_expression_correlation.txt', '_expression_correlation_by_max_abs_borzoi_bin_violinplot.pdf')
	plt.savefig(output_file)
	plt.close(fig)

	print(output_file)

	return



###########################
# Command line args
#############################
test_tissue = sys.argv[1]
borzoi_target_index = int(sys.argv[2])
borzoi_eval_output_stem = sys.argv[3]
prediction_input_data_summary_filestem = sys.argv[4]
gtex_tissue_names_file = sys.argv[5]


np.random.seed(1)

'''
# Load in all tissues names
all_tissue_names = load_in_tissue_names(gtex_tissue_names_file)

# Get index of test tissue
test_tissue_indices = np.where(all_tissue_names == test_tissue)[0]
if len(test_tissue_indices) != 1:
	print('assumption eroror')
	pdb.set_trace()
test_tissue_index = test_tissue_indices[0]




# Load in gene-based model training/evaluation data
gene_based_model_data = load_in_gene_based_model_data(prediction_input_data_summary_filestem, min_snps_per_gene=50)


tot_n_genes = len(gene_based_model_data)


evaluate_model(gene_based_model_data, test_tissue_index, borzoi_eval_output_stem, borzoi_target_index)
'''
make_expression_correlation_barplot(borzoi_eval_output_stem + '_expression_correlation.txt')
make_expression_correlation_violinplot(borzoi_eval_output_stem + '_expression_correlation.txt')
