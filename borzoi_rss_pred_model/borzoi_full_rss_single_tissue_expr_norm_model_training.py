import argparse
import numpy as np
import os
import pdb
import tensorflow as tf
import time
import re
import gzip




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

def load_in_expression_data(single_samp_per_tissue_expr_file):
	head_count = 0
	expr_mat = []
	expr_ordered_tissue_names = []
	f = open(single_samp_per_tissue_expr_file)
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			for ele in data[1:]:
				expr_ordered_tissue_names.append(ele.split(':')[1])
			continue
		expr_mat.append(np.asarray(data[1:]).astype(float))
	f.close()
	return np.asarray(expr_mat), np.asarray(expr_ordered_tissue_names)

def extract_mean_and_sdev_of_each_borzoi_feature(train_val_gene_based_model_data):
	n_genes = len(train_val_gene_based_model_data)

	for g_iter in range(n_genes):
		gene_name, gene_snp_summary_file, gene_zed_file, gene_N_eff_file, gene_LD_file, gene_inv_LD_file, gene_borzoi_pred_file, n_gene_snps = train_val_gene_based_model_data[g_iter]

		borzoi_preds = np.load(gene_borzoi_pred_file)
		valid_row_indices = np.where(~np.isnan(borzoi_preds).any(axis=1))[0]
		borzoi_preds = borzoi_preds[valid_row_indices,:]
		n_snps, n_borzoi_features = borzoi_preds.shape
		if g_iter == 0:
			counts = np.zeros(n_borzoi_features)
			sums = np.zeros(n_borzoi_features)
			sum_squares = np.zeros(n_borzoi_features)

		counts = counts + n_snps
		sums = sums + np.sum(borzoi_preds,axis=0)
		sum_squares = sum_squares + np.sum(np.square(borzoi_preds), axis=0)

	meany = sums/counts
	variance = (sum_squares - (counts*np.square(meany)))/(counts-1)
	sdev = np.sqrt(variance)

	return meany, sdev

def load_in_standardized_gene_borzoi_preds(gene_borzoi_pred_file, borzoi_feature_means, borzoi_feature_inv_sdevs):
	gene_borzoi_preds = np.load(gene_borzoi_pred_file).astype(np.float32, copy=False)
	gene_borzoi_preds -= borzoi_feature_means
	gene_borzoi_preds *= borzoi_feature_inv_sdevs
	return gene_borzoi_preds


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

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gtex-tissue-names-file', required=True, type=str)
	parser.add_argument('--prediction-input-data-summary-filestem', required=True, type=str)
	parser.add_argument('--expression-tpm-file', required=True, type=str)
	parser.add_argument('--expression-sample-file', required=True, type=str)
	parser.add_argument('--test-tissue', required=True, type=str)
	parser.add_argument('--model-training-output-stem', required=True, type=str)
	parser.add_argument('--learning-rate', required=False, type=float, default=3e-4)
	parser.add_argument('--l2-variant-reg-strength', required=False, type=float, default=1.0)
	parser.add_argument('--variant-encoder-architecture', required=False, type=str, default='128,64,32')
	return parser.parse_args()


def train_model(train_gene_based_model_data, val_gene_based_model_data, test_index, learning_rate, l2_variant_reg_strength, variant_encoder_architecture, gene_to_sdev_log_tpm, max_epochs=100, use_held_out_genes_for_validation=True):
	# Load in number data dimensions
	n_borzoi_dimensions = np.load(train_gene_based_model_data[0][6]).shape[1]

	test_tissue_indices = np.asarray([test_index])

	# Load in borzoi means and standard deviations for each feature
	borzoi_feature_means, borzoi_feature_sdevs = extract_mean_and_sdev_of_each_borzoi_feature(train_gene_based_model_data)
	borzoi_feature_means = borzoi_feature_means.astype(np.float32)
	borzoi_feature_sdevs[borzoi_feature_sdevs == 0.0] = 1.0
	borzoi_feature_inv_sdevs = 1.0*(1.0 / borzoi_feature_sdevs).astype(np.float32)

	n_training_genes = len(train_gene_based_model_data)
	best_val_loss = np.inf
	best_variant_weights = None
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	# Tissue encoder: reduces high-dimensional tissue expression to a compact embedding.
	l2_reg_variant = tf.keras.regularizers.l2(l2_variant_reg_strength)

	# Variant encoder: maps Borzoi predictions into the same low-dimensional space as tissues.
	architecture_sizes = [int(layer_size) for layer_size in variant_encoder_architecture.split(',') if layer_size != '']
	if len(architecture_sizes) < 1:
		raise ValueError('variant_encoder_architecture must contain at least one layer size')
	borzoi_input = tf.keras.Input(shape=(n_borzoi_dimensions,), name='borzoi_input')
	borzoi_hidden = borzoi_input
	for layer_num, layer_size in enumerate(architecture_sizes, start=1):
		borzoi_hidden = tf.keras.layers.Dense(layer_size, activation='relu', kernel_regularizer=l2_reg_variant, name='borzoi_dense_' + str(layer_num))(borzoi_hidden)
	predicted_variant_effect = tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=l2_reg_variant, name='predicted_variant_effect')(borzoi_hidden)
	variant_encoder = tf.keras.Model(inputs=borzoi_input, outputs=predicted_variant_effect, name='variant_effect_predictor')

	def compute_beta_mat(gene_borzoi_preds_tf, training):
		predicted_variant_effects_tf = variant_encoder(gene_borzoi_preds_tf, training=training)
		return 1e-8*predicted_variant_effects_tf


	def compute_gene_loss(gene_borzoi_preds_tf, gene_zeds_tf, gene_N_eff_tf, gene_LD_tf, gene_inv_LD_tf, gene_afs_tf, training, compute_correlation=False):
		beta_mat = compute_beta_mat(gene_borzoi_preds_tf, training=training)
		genotype_sd = tf.sqrt(2.0 * gene_afs_tf * (1.0 - gene_afs_tf))
		beta_std_mat = beta_mat * genotype_sd[:, None]


		pred_gene_zeds_mat = tf.sqrt(gene_N_eff_tf) * tf.matmul(gene_LD_tf, beta_std_mat)
		residuals = gene_zeds_tf - pred_gene_zeds_mat

		loss = tf.reduce_sum(residuals * tf.matmul(gene_inv_LD_tf, residuals))

		if compute_correlation:
			obs = tf.reshape(gene_zeds_tf, [-1])
			pred = tf.reshape(pred_gene_zeds_mat, [-1])

			obs_centered = obs - tf.reduce_mean(obs)
			pred_centered = pred - tf.reduce_mean(pred)

			corr = tf.reduce_sum(obs_centered * pred_centered) / (
    		tf.sqrt(tf.reduce_sum(tf.square(obs_centered))) *
    		tf.sqrt(tf.reduce_sum(tf.square(pred_centered)))
			)
			return loss, corr
		else:
			return loss

	'''
	@tf.function(
		input_signature=[
			tf.TensorSpec(shape=[None, n_borzoi_dimensions], dtype=tf.float32),
			tf.TensorSpec(shape=[None, None], dtype=tf.float32),
			tf.TensorSpec(shape=[None, None], dtype=tf.float32),
			tf.TensorSpec(shape=[None, None], dtype=tf.float32),
			tf.TensorSpec(shape=[None], dtype=tf.float32),
		],
		reduce_retracing=True
	)
	'''
	def train_step(gene_borzoi_preds_tf, gene_zeds_tf, gene_N_eff_tf, gene_LD_tf, gene_inv_LD_tf, gene_afs_tf):
		with tf.GradientTape() as tape:
			train_loss = compute_gene_loss(
				gene_borzoi_preds_tf,
				gene_zeds_tf,
				gene_N_eff_tf,
				gene_LD_tf,
				gene_inv_LD_tf,
				gene_afs_tf,
				training=True,
				compute_correlation=False
			)
		trainable_variables = variant_encoder.trainable_variables
		gradients = tape.gradient(train_loss, trainable_variables)
		optimizer.apply_gradients(zip(gradients, trainable_variables))
		return train_loss

	# Epoch training
	for epoch_iter in range(max_epochs):
		epoch_train_losses = []

		# Loop through genes
		for gene_index in np.random.permutation(n_training_genes):
			# Load in training data for this gene
			gene_name, gene_snp_summary_file, gene_zed_file, gene_N_eff_file, gene_LD_file, gene_inv_LD_file, gene_borzoi_pred_file, n_gene_snps = train_gene_based_model_data[gene_index]
			gene_LD = np.load(gene_LD_file)
			gene_inv_LD = np.load(gene_inv_LD_file)
			gene_borzoi_preds = load_in_standardized_gene_borzoi_preds(gene_borzoi_pred_file, borzoi_feature_means, borzoi_feature_inv_sdevs)
			gene_zeds = np.load(gene_zed_file)
			gene_N_eff = np.load(gene_N_eff_file)
			gene_afs = np.loadtxt(gene_snp_summary_file,dtype=str)[1:,-1].astype(float)
			valid_row_indices = np.where(~np.isnan(gene_borzoi_preds).any(axis=1))[0]
			if len(valid_row_indices) == 0:
				continue
			if gene_name not in gene_to_sdev_log_tpm:
				continue
			gene_sdev = gene_to_sdev_log_tpm[gene_name]
			if gene_sdev == 0:
				print('sdev equal zero error')
				continue

			gene_LD = gene_LD[valid_row_indices, :][:, valid_row_indices]
			gene_borzoi_preds = gene_borzoi_preds[valid_row_indices, :]
			gene_N_eff = gene_N_eff[valid_row_indices, :][:, test_tissue_indices]
			gene_N_eff = gene_N_eff/np.square(gene_sdev)
			gene_zeds = gene_zeds[valid_row_indices, :][:, test_tissue_indices]
			gene_afs = gene_afs[valid_row_indices]

			if gene_LD.shape[0] != gene_inv_LD.shape[0]:
				print('assumption erroror')
				pdb.set_trace()
			if np.sum(np.isnan(gene_zeds)) > 0:
				continue

			gene_LD_tf = tf.convert_to_tensor(gene_LD.astype(np.float32))
			gene_inv_LD_tf = tf.convert_to_tensor(gene_inv_LD.astype(np.float32))
			gene_borzoi_preds_tf = tf.convert_to_tensor(gene_borzoi_preds.astype(np.float32))
			gene_N_eff_tf = tf.convert_to_tensor(gene_N_eff.astype(np.float32))
			gene_zeds_tf = tf.convert_to_tensor(gene_zeds.astype(np.float32))
			gene_afs_tf = tf.convert_to_tensor(gene_afs.astype(np.float32))
			train_loss = train_step(gene_borzoi_preds_tf, gene_zeds_tf, gene_N_eff_tf, gene_LD_tf, gene_inv_LD_tf, gene_afs_tf)
			epoch_train_losses.append(train_loss.numpy())

		epoch_val_losses = []
		epoch_val_corrs = []
		for gene_name, gene_snp_summary_file, gene_zed_file, gene_N_eff_file, gene_LD_file, gene_inv_LD_file, gene_borzoi_pred_file, n_gene_snps in val_gene_based_model_data:
			gene_LD = np.load(gene_LD_file)
			gene_inv_LD = np.load(gene_inv_LD_file)
			gene_borzoi_preds = load_in_standardized_gene_borzoi_preds(gene_borzoi_pred_file, borzoi_feature_means, borzoi_feature_inv_sdevs)
			gene_zeds = np.load(gene_zed_file)
			gene_N_eff = np.load(gene_N_eff_file)
			gene_afs = np.loadtxt(gene_snp_summary_file,dtype=str)[1:,-1].astype(float)
			valid_row_indices = np.where(~np.isnan(gene_borzoi_preds).any(axis=1))[0]
			if len(valid_row_indices) == 0:
				continue
			if gene_name not in gene_to_sdev_log_tpm:
				continue
			gene_sdev = gene_to_sdev_log_tpm[gene_name]
			if gene_sdev == 0.0:
				continue


			gene_LD = gene_LD[valid_row_indices, :][:, valid_row_indices]
			gene_borzoi_preds = gene_borzoi_preds[valid_row_indices, :]
			gene_N_eff = gene_N_eff[valid_row_indices, :][:, test_tissue_indices]
			gene_zeds = gene_zeds[valid_row_indices, :][:, test_tissue_indices]
			gene_afs = gene_afs[valid_row_indices]

			gene_N_eff = gene_N_eff/np.square(gene_sdev)

			if np.sum(np.isnan(gene_zeds)) > 0:
				continue
			if len(valid_row_indices) == 0:
				continue

			gene_LD_tf = tf.convert_to_tensor(gene_LD.astype(np.float32))
			gene_inv_LD_tf = tf.convert_to_tensor(gene_inv_LD.astype(np.float32))
			gene_borzoi_preds_tf = tf.convert_to_tensor(gene_borzoi_preds.astype(np.float32))
			gene_N_eff_tf = tf.convert_to_tensor(gene_N_eff.astype(np.float32))
			gene_zeds_tf = tf.convert_to_tensor(gene_zeds.astype(np.float32))
			gene_afs_tf = tf.convert_to_tensor(gene_afs.astype(np.float32))
			val_loss, val_corr = compute_gene_loss(
				gene_borzoi_preds_tf,
				gene_zeds_tf,
				gene_N_eff_tf,
				gene_LD_tf,
				gene_inv_LD_tf,
				gene_afs_tf,
				training=False,
				compute_correlation=True
			)
			epoch_val_corrs.append(val_corr.numpy())
			epoch_val_losses.append(val_loss.numpy())

		epoch_train_losses = np.asarray(epoch_train_losses)
		epoch_val_losses = np.asarray(epoch_val_losses)
		epoch_val_corrs = np.asarray(epoch_val_corrs)
		epoch_train_losses = epoch_train_losses[np.isnan(epoch_train_losses) == False]
		epoch_val_losses = epoch_val_losses[np.isnan(epoch_val_losses) == False]
		epoch_val_corrs = epoch_val_corrs[np.isnan(epoch_val_corrs) == False]
		epoch_val_loss = np.mean(epoch_val_losses)
		status = ''
		if epoch_val_loss < best_val_loss:
			best_val_loss = epoch_val_loss
			best_variant_weights = variant_encoder.get_weights()
			status = ' best'
		print('epoch ' + str(epoch_iter) + ' train_loss=' + str(np.mean(epoch_train_losses)) + ' val_loss=' + str(epoch_val_loss) + ' val_corr=' + str(np.mean(epoch_val_corrs)) + status, flush=True)


	if best_variant_weights is not None:
		variant_encoder.set_weights(best_variant_weights)

	return variant_encoder


def evaluate_model(variant_encoder, gene_based_model_data, train_gene_based_model_data, val_gene_based_model_data, test_gene_based_model_data, test_tissue_index, gene_to_sdev_log_tpm, model_training_output_stem):
	train_gene_names = {gene_data[0] for gene_data in train_gene_based_model_data}
	val_gene_names = {gene_data[0] for gene_data in val_gene_based_model_data}
	test_gene_names = {gene_data[0] for gene_data in test_gene_based_model_data}

	borzoi_feature_means, borzoi_feature_sdevs = extract_mean_and_sdev_of_each_borzoi_feature(train_gene_based_model_data)
	borzoi_feature_means = borzoi_feature_means.astype(np.float32)
	borzoi_feature_sdevs[borzoi_feature_sdevs == 0.0] = 1.0
	borzoi_feature_inv_sdevs = 1.0*(1.0 / borzoi_feature_sdevs).astype(np.float32)

	output_file = model_training_output_stem + '_all_gene_test_tissue_evaluation.txt'
	variant_output_file = model_training_output_stem + '_all_variant_gene_pairs_test_tissue_evaluation.txt'

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

			if gene_name not in gene_to_sdev_log_tpm:
				continue
			gene_sdev = gene_to_sdev_log_tpm[gene_name]
			if gene_sdev == 0.0:
				continue

			gene_LD = np.load(gene_LD_file)
			gene_inv_LD = np.load(gene_inv_LD_file)
			gene_borzoi_preds = load_in_standardized_gene_borzoi_preds(gene_borzoi_pred_file, borzoi_feature_means, borzoi_feature_inv_sdevs)
			gene_zeds = np.load(gene_zed_file)
			gene_N_eff = np.load(gene_N_eff_file)
			gene_snp_summary = np.loadtxt(gene_snp_summary_file, dtype=str)
			gene_variant_names = gene_snp_summary[1:, 0]
			gene_afs = gene_snp_summary[1:, -1].astype(float)
			valid_row_indices = np.where(~np.isnan(gene_borzoi_preds).any(axis=1))[0]
			if len(valid_row_indices) == 0:
				t.write(gene_name + '\t' + gene_split + '\t' + str(n_gene_snps) + '\tnan\tnan\tnan\n')
				continue

			gene_LD = gene_LD[valid_row_indices, :][:, valid_row_indices]
			gene_borzoi_preds = gene_borzoi_preds[valid_row_indices, :]
			gene_N_eff = gene_N_eff[valid_row_indices, :][:, [test_tissue_index]]
			gene_N_eff = gene_N_eff/np.square(gene_sdev)
			gene_zeds = gene_zeds[valid_row_indices, :][:, [test_tissue_index]]
			gene_variant_names = gene_variant_names[valid_row_indices]
			gene_afs = gene_afs[valid_row_indices]

			gene_LD_tf = tf.convert_to_tensor(gene_LD.astype(np.float32))
			gene_inv_LD_tf = tf.convert_to_tensor(gene_inv_LD.astype(np.float32))
			gene_borzoi_preds_tf = tf.convert_to_tensor(gene_borzoi_preds.astype(np.float32))
			gene_N_eff_tf = tf.convert_to_tensor(gene_N_eff.astype(np.float32))
			gene_zeds_tf = tf.convert_to_tensor(gene_zeds.astype(np.float32))
			gene_afs_tf = tf.convert_to_tensor(gene_afs.astype(np.float32))

			predicted_variant_effects_tf = variant_encoder(gene_borzoi_preds_tf, training=False)
			beta_mat = 1e-8*predicted_variant_effects_tf
			genotype_sd = tf.sqrt(2.0 * gene_afs_tf * (1.0 - gene_afs_tf))
			beta_std_mat = beta_mat * genotype_sd[:, None]
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


def to_underscore(s: str) -> str:
	s = s.replace(")", "")
	# replace " (" OR any run of spaces and hyphens with "_"
	return re.sub(r"(?:\s*\(|[ -]+)", "_", s)


def create_mapping_from_gene_name_to_sdev_log_tpm(gtex_expression_tpm_file, gtex_expression_sample_file, test_tissue):
	# First create dictionary list of rna-seq samples in this tissue
	gtex_samples = {}
	f = open(gtex_expression_sample_file)
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		gtex_sample_id = data[0]
		tissue_type = to_underscore(data[6])
		if tissue_type == 'Cells_EBV_transformed_lymphocytes':
			tissue_type = 'Cells_EBV-transformed_lymphocytes'
		elif tissue_type == 'Brain_Spinal_cord_cervical_c_1':
			tissue_type = 'Brain_Spinal_cord_cervical_c-1'

		if tissue_type != test_tissue:
			continue
		gtex_samples[gtex_sample_id] = 1
	f.close()
	print(len(gtex_samples))

	# Second create mapping from gene name to sdev(log_tpm)
	gene_to_sdev_dicti = {}
	f = gzip.open(gtex_expression_tpm_file,'rt')
	counter = 0
	head_count = -1
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		head_count = head_count + 1
		if head_count < 2:
			continue
		if head_count == 2: # header
			indices = []
			for ii, ele in enumerate(data):
				if ele in gtex_samples:
					indices.append(ii)
			indices = np.asarray(indices)
			continue
		ens_id = data[0].split('.')[0]
		expr_samples = (np.asarray(data)[indices]).astype(float)
		log_expr = np.log(expr_samples + 1.0)

		gene_to_sdev_dicti[ens_id] = np.std(log_expr,ddof=1)
		counter = counter + 1

	f.close()

	return gene_to_sdev_dicti


def main():
	args = parse_args()

	###########################
	# Load in data
	############################
	gtex_tissue_names_file = args.gtex_tissue_names_file
	prediction_input_data_summary_filestem = args.prediction_input_data_summary_filestem
	test_tissue = args.test_tissue
	model_training_output_stem = args.model_training_output_stem
	learning_rate = args.learning_rate
	l2_variant_reg_strength = args.l2_variant_reg_strength
	variant_encoder_architecture = args.variant_encoder_architecture
	gtex_expression_tpm_file = args.expression_tpm_file
	gtex_expression_sample_file = args.expression_sample_file

	np.random.seed(1)

	gene_to_sdev_log_tpm = create_mapping_from_gene_name_to_sdev_log_tpm(gtex_expression_tpm_file, gtex_expression_sample_file, test_tissue)


	# Load in all tissues names
	all_tissue_names = load_in_tissue_names(gtex_tissue_names_file)

	# Get index of test tissue
	test_tissue_indices = np.where(all_tissue_names == test_tissue)[0]
	if len(test_tissue_indices) != 1:
		print('assumption eroror')
		pdb.set_trace()
	test_tissue_index = test_tissue_indices[0]


	###############################################
	# DELTE LATER ***** JUST USED FOR RANDOM SEED
	# Does nothing other than ensures comparable random seed
	n_val_tissues = 5
	train_val_tissue_indices = np.arange(len(all_tissue_names))
	train_val_tissue_indices = np.delete(train_val_tissue_indices, test_tissue_index)
	val_tissue_indices = np.sort(np.random.choice(train_val_tissue_indices, size=n_val_tissues, replace=False))
	train_tissue_indices = train_val_tissue_indices[np.isin(train_val_tissue_indices, val_tissue_indices, invert=True)]
	train_tissue_names = all_tissue_names[train_tissue_indices]
	val_tissue_names = all_tissue_names[val_tissue_indices]
	#############################################


	# Load in gene-based model training/evaluation data
	gene_based_model_data = load_in_gene_based_model_data(prediction_input_data_summary_filestem, min_snps_per_gene=50)
	tot_n_genes = len(gene_based_model_data)
	n_train_val_genes = int(np.floor(tot_n_genes*.8))
	# Split into train/val and test
	train_val_gene_based_model_data = gene_based_model_data[:n_train_val_genes]
	test_gene_based_model_data = gene_based_model_data[n_train_val_genes:]


	###########################
	# Ready for model training
	############################
	max_epochs=40
	use_held_out_genes_for_validation=True
	train_gene_based_model_data, val_gene_based_model_data = split_train_and_val_gene_based_model_data(train_val_gene_based_model_data, use_held_out_genes_for_validation)
	# Train
	variant_encoder = train_model(
		train_gene_based_model_data,
		val_gene_based_model_data,
		test_tissue_index,
		learning_rate,
		l2_variant_reg_strength,
		variant_encoder_architecture,
		gene_to_sdev_log_tpm,
		max_epochs=max_epochs,
		use_held_out_genes_for_validation=use_held_out_genes_for_validation
	)
	# Evaluate
	evaluate_model(variant_encoder, gene_based_model_data, train_gene_based_model_data, val_gene_based_model_data, test_gene_based_model_data, test_tissue_index, gene_to_sdev_log_tpm, model_training_output_stem)
	# Save model results
	variant_encoder.save(model_training_output_stem + '_variant_encoder.keras')




if __name__ == "__main__":
	main()
