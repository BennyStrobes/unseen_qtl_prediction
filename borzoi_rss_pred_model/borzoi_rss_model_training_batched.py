import argparse
import numpy as np
import os
import pdb
import tensorflow as tf
import time









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
			borzoi_file = data[5]
			n_snps_per_gene = int(data[7])
			if n_snps_per_gene < min_snps_per_gene:
				continue
			counter = counter + 1

			if np.random.choice(np.arange(20)) == 1:
				arr.append((gene_name, snp_summary_file, zed_file, N_eff_file, ld_file, borzoi_file, n_snps_per_gene))
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
		gene_name, gene_snp_summary_file, gene_zed_file, gene_N_eff_file, gene_LD_file, gene_borzoi_pred_file, n_gene_snps = train_val_gene_based_model_data[g_iter]

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
	parser.add_argument('--single-samp-per-tissue-expr-file', required=True, type=str)
	parser.add_argument('--prediction-input-data-summary-filestem', required=True, type=str)
	parser.add_argument('--test-tissue', required=True, type=str)
	parser.add_argument('--model-training-output-stem', required=True, type=str)
	parser.add_argument('--n-val-tissues', required=False, type=int, default=5)
	parser.add_argument('--learning-rate', required=False, type=float, default=3e-4)
	parser.add_argument('--l2-tissue-reg-strength', required=False, type=float, default=1e-5)
	parser.add_argument('--l2-variant-reg-strength', required=False, type=float, default=1.0)
	parser.add_argument('--variant-encoder-architecture', required=False, type=str, default='128,64,32')
	parser.add_argument('--dropout-rate', required=False, type=float, default=0.0)
	return parser.parse_args()


def train_model(train_gene_based_model_data, val_gene_based_model_data, gene_expression_data, train_tissue_indices, val_tissue_indices, learning_rate, l2_tissue_reg_strength, l2_variant_reg_strength, variant_encoder_architecture, dropout_rate, max_epochs=100, use_held_out_genes_for_validation=True):
	genes_per_gradient_step = 5
	# Load in number data dimensions
	n_borzoi_dimensions = np.load(train_gene_based_model_data[0][5]).shape[1]
	n_expr_gene_dimensions = gene_expression_data.shape[0]

	# Load in borzoi means and standard deviations for each feature
	borzoi_feature_means, borzoi_feature_sdevs = extract_mean_and_sdev_of_each_borzoi_feature(train_gene_based_model_data)
	borzoi_feature_means = borzoi_feature_means.astype(np.float32)
	borzoi_feature_sdevs[borzoi_feature_sdevs == 0.0] = 1.0
	borzoi_feature_inv_sdevs = (1.0 / borzoi_feature_sdevs).astype(np.float32)

	n_training_genes = len(train_gene_based_model_data)
	best_val_loss = np.inf
	best_tissue_weights = None
	best_variant_weights = None
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	# Tissue encoder: reduces high-dimensional tissue expression to a compact embedding.
	l2_reg_tissue = tf.keras.regularizers.l2(l2_tissue_reg_strength)
	l2_reg_variant = tf.keras.regularizers.l2(l2_variant_reg_strength)
	tissue_input = tf.keras.Input(shape=(n_expr_gene_dimensions,), name='tissue_expression_input')
	tissue_hidden = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2_reg_tissue, name='tissue_dense_1')(tissue_input)
	tissue_hidden = tf.keras.layers.Dropout(dropout_rate, name='tissue_dropout_1')(tissue_hidden)
	tissue_hidden = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2_reg_tissue, name='tissue_dense_2')(tissue_hidden)
	tissue_hidden = tf.keras.layers.Dropout(dropout_rate, name='tissue_dropout_2')(tissue_hidden)
	tissue_embedding = tf.keras.layers.Dense(32, activation='linear', kernel_regularizer=l2_reg_tissue, name='tissue_embedding')(tissue_hidden)
	tissue_encoder = tf.keras.Model(inputs=tissue_input, outputs=tissue_embedding, name='tissue_encoder')

	# Variant encoder: maps Borzoi predictions into the same low-dimensional space as tissues.
	architecture_sizes = [int(layer_size) for layer_size in variant_encoder_architecture.split(',') if layer_size != '']
	if len(architecture_sizes) < 1:
		raise ValueError('variant_encoder_architecture must contain at least one layer size')
	borzoi_input = tf.keras.Input(shape=(n_borzoi_dimensions,), name='borzoi_input')
	borzoi_hidden = borzoi_input
	for layer_num, layer_size in enumerate(architecture_sizes[:-1], start=1):
		borzoi_hidden = tf.keras.layers.Dense(layer_size, activation='relu', kernel_regularizer=l2_reg_variant, name='borzoi_dense_' + str(layer_num))(borzoi_hidden)
		borzoi_hidden = tf.keras.layers.Dropout(dropout_rate, name='borzoi_dropout_' + str(layer_num))(borzoi_hidden)
	variant_embedding = tf.keras.layers.Dense(architecture_sizes[-1], activation='linear', kernel_regularizer=l2_reg_variant, name='variant_embedding')(borzoi_hidden)
	variant_encoder = tf.keras.Model(inputs=borzoi_input, outputs=variant_embedding, name='variant_encoder')

	train_tissue_expression_tf = tf.convert_to_tensor(gene_expression_data[:, train_tissue_indices].T.astype(np.float32))
	val_tissue_expression_tf = tf.convert_to_tensor(gene_expression_data[:, val_tissue_indices].T.astype(np.float32))

	def compute_beta_mat(gene_borzoi_preds_tf, tissue_expression_tf, training):
		tissue_embeddings_tf = tissue_encoder(tissue_expression_tf, training=training)
		variant_embeddings_tf = variant_encoder(gene_borzoi_preds_tf, training=training)
		return tf.matmul(variant_embeddings_tf, tissue_embeddings_tf, transpose_b=True)

	def compute_observed_and_predicted_gene_zeds(gene_borzoi_preds_tf, gene_zeds_tf, gene_N_eff_tf, gene_LD_tf, tissue_expression_tf, gene_afs_tf, training):
		beta_mat = compute_beta_mat(gene_borzoi_preds_tf, tissue_expression_tf, training=training)*1e-8
		valid_mask = ~tf.math.is_nan(gene_zeds_tf)
		genotype_sd = tf.sqrt(2.0 * gene_afs_tf * (1.0 - gene_afs_tf))
		beta_std_mat = beta_mat * genotype_sd[:, None]
		pred_gene_zeds = tf.sqrt(gene_N_eff_tf[valid_mask]) * (tf.matmul(gene_LD_tf, beta_std_mat))[valid_mask]
		obs_gene_zeds = gene_zeds_tf[valid_mask]
		return obs_gene_zeds, pred_gene_zeds, valid_mask

	def compute_gene_loss(gene_borzoi_preds_tf, gene_zeds_tf, gene_N_eff_tf, gene_LD_tf, tissue_expression_tf, gene_afs_tf, training):
		if tissue_expression_tf.shape[0] == 0:
			return tf.constant(0.0, dtype=tf.float32)

		obs_gene_zeds, pred_gene_zeds, valid_mask = compute_observed_and_predicted_gene_zeds(
			gene_borzoi_preds_tf,
			gene_zeds_tf,
			gene_N_eff_tf,
			gene_LD_tf,
			tissue_expression_tf,
			gene_afs_tf,
			training
		)
		residuals = obs_gene_zeds - pred_gene_zeds

		ld_scores = tf.reduce_sum(tf.square(gene_LD_tf), axis=1)
		ld_scores_mat = tf.broadcast_to(ld_scores[:, None], tf.shape(gene_zeds_tf))  # (n_snps, n_tissues)
		valid_ld_scores = ld_scores_mat[valid_mask]  

		#effective_num_snps = tf.cast(tf.shape(ld_scores)[0], tf.float32) / tf.reduce_mean(ld_scores)
		#effective_num_obs = tf.reduce_sum(tf.cast(valid_mask, tf.float32))/tf.reduce_mean(ld_scores)

		return tf.reduce_sum(tf.square(residuals)/valid_ld_scores)

	def load_gene_training_tensors(gene_data, tissue_indices):
		gene_name, gene_snp_summary_file, gene_zed_file, gene_N_eff_file, gene_LD_file, gene_borzoi_pred_file, n_gene_snps = gene_data
		gene_LD = np.load(gene_LD_file)
		gene_borzoi_preds = load_in_standardized_gene_borzoi_preds(gene_borzoi_pred_file, borzoi_feature_means, borzoi_feature_inv_sdevs)
		gene_zeds = np.load(gene_zed_file)
		gene_N_eff = np.load(gene_N_eff_file)
		gene_afs = np.loadtxt(gene_snp_summary_file,dtype=str)[1:,-1].astype(float)
		valid_row_indices = np.where(~np.isnan(gene_borzoi_preds).any(axis=1))[0]
		if len(valid_row_indices) == 0:
			return None
		gene_LD = gene_LD[valid_row_indices, :][:, valid_row_indices]
		gene_borzoi_preds = gene_borzoi_preds[valid_row_indices, :]
		gene_N_eff = gene_N_eff[valid_row_indices, :][:, tissue_indices]
		gene_zeds = gene_zeds[valid_row_indices, :][:, tissue_indices]
		gene_afs = gene_afs[valid_row_indices]
		return (
			tf.convert_to_tensor(gene_borzoi_preds.astype(np.float32)),
			tf.convert_to_tensor(gene_zeds.astype(np.float32)),
			tf.convert_to_tensor(gene_N_eff.astype(np.float32)),
			tf.convert_to_tensor(gene_LD.astype(np.float32)),
			tf.convert_to_tensor(gene_afs.astype(np.float32))
		)

	def train_step(gene_batch_tensors):
		with tf.GradientTape() as tape:
			batch_losses = []
			for gene_borzoi_preds_tf, gene_zeds_tf, gene_N_eff_tf, gene_LD_tf, gene_afs_tf in gene_batch_tensors:
				batch_losses.append(compute_gene_loss(
					gene_borzoi_preds_tf,
					gene_zeds_tf,
					gene_N_eff_tf,
					gene_LD_tf,
					train_tissue_expression_tf,
					gene_afs_tf,
					training=True
				))
			train_loss = tf.add_n(batch_losses) / tf.cast(len(batch_losses), tf.float32)
		trainable_variables = tissue_encoder.trainable_variables + variant_encoder.trainable_variables
		gradients = tape.gradient(train_loss, trainable_variables)
		optimizer.apply_gradients(zip(gradients, trainable_variables))
		return train_loss

	# Epoch training
	for epoch_iter in range(max_epochs):
		epoch_train_losses = []

		shuffled_gene_indices = np.random.permutation(n_training_genes)
		for batch_start in range(0, n_training_genes, genes_per_gradient_step):
			gene_batch_tensors = []
			for gene_index in shuffled_gene_indices[batch_start:(batch_start + genes_per_gradient_step)]:
				gene_tensors = load_gene_training_tensors(train_gene_based_model_data[gene_index], train_tissue_indices)
				if gene_tensors is not None:
					gene_batch_tensors.append(gene_tensors)
			if len(gene_batch_tensors) == 0:
				continue
			train_loss = train_step(gene_batch_tensors)
			epoch_train_losses.append(train_loss.numpy())

		epoch_val_losses = []
		epoch_val_corrs = []
		for gene_name, gene_snp_summary_file, gene_zed_file, gene_N_eff_file, gene_LD_file, gene_borzoi_pred_file, n_gene_snps in val_gene_based_model_data:
			gene_LD = np.load(gene_LD_file)
			gene_borzoi_preds = load_in_standardized_gene_borzoi_preds(gene_borzoi_pred_file, borzoi_feature_means, borzoi_feature_inv_sdevs)
			gene_zeds = np.load(gene_zed_file)
			gene_N_eff = np.load(gene_N_eff_file)
			gene_afs = np.loadtxt(gene_snp_summary_file,dtype=str)[1:,-1].astype(float)
			valid_row_indices = np.where(~np.isnan(gene_borzoi_preds).any(axis=1))[0]
			if len(valid_row_indices) == 0:
				continue
			gene_LD = gene_LD[valid_row_indices, :][:, valid_row_indices]
			gene_borzoi_preds = gene_borzoi_preds[valid_row_indices, :]
			gene_N_eff = gene_N_eff[valid_row_indices, :][:, val_tissue_indices]
			gene_zeds = gene_zeds[valid_row_indices, :][:, val_tissue_indices]
			gene_afs = gene_afs[valid_row_indices]
			gene_LD_tf = tf.convert_to_tensor(gene_LD.astype(np.float32))
			gene_borzoi_preds_tf = tf.convert_to_tensor(gene_borzoi_preds.astype(np.float32))
			gene_N_eff_tf = tf.convert_to_tensor(gene_N_eff.astype(np.float32))
			gene_zeds_tf = tf.convert_to_tensor(gene_zeds.astype(np.float32))
			gene_afs_tf = tf.convert_to_tensor(gene_afs.astype(np.float32))
			obs_gene_zeds, pred_gene_zeds, valid_mask  = compute_observed_and_predicted_gene_zeds(
				gene_borzoi_preds_tf,
				gene_zeds_tf,
				gene_N_eff_tf,
				gene_LD_tf,
				val_tissue_expression_tf,
				gene_afs_tf,
				training=False
			)
			residuals = obs_gene_zeds - pred_gene_zeds
			ld_scores = tf.reduce_sum(tf.square(gene_LD_tf), axis=1)
			ld_scores_mat = tf.broadcast_to(ld_scores[:, None], tf.shape(gene_zeds_tf))
			valid_ld_scores = ld_scores_mat[valid_mask]
			val_loss = tf.reduce_sum(tf.square(residuals) / valid_ld_scores)

			if tf.size(obs_gene_zeds) > 1:
				gene_corr = np.corrcoef(obs_gene_zeds.numpy(), pred_gene_zeds.numpy())[0, 1]
				epoch_val_corrs.append(gene_corr)
			else:
				epoch_val_corrs.append(np.nan)
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
			best_tissue_weights = tissue_encoder.get_weights()
			best_variant_weights = variant_encoder.get_weights()
			status = ' best'
		print('epoch ' + str(epoch_iter) + ' train_loss=' + str(np.mean(epoch_train_losses)) + ' val_loss=' + str(epoch_val_loss) + ' val_corr=' + str(np.mean(epoch_val_corrs)) + status, flush=True)


	if best_tissue_weights is not None:
		tissue_encoder.set_weights(best_tissue_weights)
		variant_encoder.set_weights(best_variant_weights)

	return tissue_encoder, variant_encoder


def evaluate_model(tissue_encoder, variant_encoder, gene_based_model_data, train_gene_based_model_data, val_gene_based_model_data, test_gene_based_model_data, gene_expression_data, test_tissue_index, model_training_output_stem):
	train_gene_names = {gene_data[0] for gene_data in train_gene_based_model_data}
	val_gene_names = {gene_data[0] for gene_data in val_gene_based_model_data}
	test_gene_names = {gene_data[0] for gene_data in test_gene_based_model_data}

	borzoi_feature_means, borzoi_feature_sdevs = extract_mean_and_sdev_of_each_borzoi_feature(train_gene_based_model_data)
	borzoi_feature_means = borzoi_feature_means.astype(np.float32)
	borzoi_feature_sdevs[borzoi_feature_sdevs == 0.0] = 1.0
	borzoi_feature_inv_sdevs = (1.0 / borzoi_feature_sdevs).astype(np.float32)

	test_tissue_expression_tf = tf.convert_to_tensor(gene_expression_data[:, [test_tissue_index]].T.astype(np.float32))
	output_file = model_training_output_stem + '_all_gene_test_tissue_evaluation.txt'

	with open(output_file, 'w') as t:
		t.write('gene_name\tgene_split\tn_snps\tloss\tcorr\tpred_expr_corr\n')
		for gene_name, gene_snp_summary_file, gene_zed_file, gene_N_eff_file, gene_LD_file, gene_borzoi_pred_file, n_gene_snps in gene_based_model_data:
			if gene_name in train_gene_names:
				gene_split = 'train'
			elif gene_name in val_gene_names:
				gene_split = 'validation'
			elif gene_name in test_gene_names:
				gene_split = 'test'
			else:
				gene_split = 'unknown'

			gene_LD = np.load(gene_LD_file)
			gene_borzoi_preds = load_in_standardized_gene_borzoi_preds(gene_borzoi_pred_file, borzoi_feature_means, borzoi_feature_inv_sdevs)
			gene_zeds = np.load(gene_zed_file)
			gene_N_eff = np.load(gene_N_eff_file)
			gene_afs = np.loadtxt(gene_snp_summary_file, dtype=str)[1:, -1].astype(float)
			valid_row_indices = np.where(~np.isnan(gene_borzoi_preds).any(axis=1))[0]
			if len(valid_row_indices) == 0:
				t.write(gene_name + '\t' + gene_split + '\t' + str(n_gene_snps) + '\tnan\tnan\tnan\n')
				continue

			gene_LD = gene_LD[valid_row_indices, :][:, valid_row_indices]
			gene_borzoi_preds = gene_borzoi_preds[valid_row_indices, :]
			gene_N_eff = gene_N_eff[valid_row_indices, :][:, [test_tissue_index]]
			gene_zeds = gene_zeds[valid_row_indices, :][:, [test_tissue_index]]
			gene_afs = gene_afs[valid_row_indices]

			gene_LD_tf = tf.convert_to_tensor(gene_LD.astype(np.float32))
			gene_borzoi_preds_tf = tf.convert_to_tensor(gene_borzoi_preds.astype(np.float32))
			gene_N_eff_tf = tf.convert_to_tensor(gene_N_eff.astype(np.float32))
			gene_zeds_tf = tf.convert_to_tensor(gene_zeds.astype(np.float32))
			gene_afs_tf = tf.convert_to_tensor(gene_afs.astype(np.float32))

			tissue_embeddings_tf = tissue_encoder(test_tissue_expression_tf, training=False)
			variant_embeddings_tf = variant_encoder(gene_borzoi_preds_tf, training=False)
			beta_mat = tf.matmul(variant_embeddings_tf, tissue_embeddings_tf, transpose_b=True)*1e-8
			genotype_sd = tf.sqrt(2.0 * gene_afs_tf * (1.0 - gene_afs_tf))
			beta_std_mat = beta_mat * genotype_sd[:, None]
			valid_mask = ~tf.math.is_nan(gene_zeds_tf)
			pred_gene_zeds = tf.sqrt(gene_N_eff_tf[valid_mask]) * (tf.matmul(gene_LD_tf, beta_std_mat))[valid_mask]
			obs_gene_zeds = gene_zeds_tf[valid_mask]
			residuals = obs_gene_zeds - pred_gene_zeds
			ld_scores = tf.reduce_sum(tf.square(gene_LD_tf), axis=1)
			ld_scores_mat = tf.broadcast_to(ld_scores[:, None], tf.shape(gene_zeds_tf))
			valid_ld_scores = tf.maximum(ld_scores_mat[valid_mask], 1e-8)
			gene_loss = tf.reduce_sum(tf.square(residuals) / valid_ld_scores).numpy()

			if tf.size(obs_gene_zeds) > 1:
				gene_corr = np.corrcoef(obs_gene_zeds.numpy(), pred_gene_zeds.numpy())[0, 1]
				std_beta = obs_gene_zeds.numpy()/(np.sqrt(gene_N_eff_tf.numpy())[:,0])
				est_causal_effects = beta_std_mat.numpy()[:,0]
				expr_corr = np.dot(std_beta, est_causal_effects)/np.sqrt(np.dot(np.dot(est_causal_effects, gene_LD),est_causal_effects))
			else:
				gene_corr = np.nan
				expr_corr = np.nan

			t.write(gene_name + '\t' + gene_split + '\t' + str(n_gene_snps) + '\t' + str(gene_loss) + '\t' + str(gene_corr) + '\t' + str(expr_corr) + '\n')


def main():
	args = parse_args()

	###########################
	# Load in data
	############################
	gtex_tissue_names_file = args.gtex_tissue_names_file
	single_samp_per_tissue_expr_file = args.single_samp_per_tissue_expr_file
	prediction_input_data_summary_filestem = args.prediction_input_data_summary_filestem
	test_tissue = args.test_tissue
	model_training_output_stem = args.model_training_output_stem
	n_val_tissues = args.n_val_tissues
	learning_rate = args.learning_rate
	l2_tissue_reg_strength = args.l2_tissue_reg_strength
	l2_variant_reg_strength = args.l2_variant_reg_strength
	variant_encoder_architecture = args.variant_encoder_architecture
	dropout_rate = args.dropout_rate

	np.random.seed(3)

	# Load in all tissues names
	all_tissue_names = load_in_tissue_names(gtex_tissue_names_file)

	# Get index of test tissue
	test_tissue_indices = np.where(all_tissue_names == test_tissue)[0]
	if len(test_tissue_indices) != 1:
		print('assumption eroror')
		pdb.set_trace()
	test_tissue_index = test_tissue_indices[0]

	# Get indices of training + val tissue
	train_val_tissue_indices = np.arange(len(all_tissue_names))
	train_val_tissue_indices = np.delete(train_val_tissue_indices, test_tissue_index)
	val_tissue_indices = np.sort(np.random.choice(train_val_tissue_indices, size=n_val_tissues, replace=False))
	train_tissue_indices = train_val_tissue_indices[np.isin(train_val_tissue_indices, val_tissue_indices, invert=True)]
	train_tissue_names = all_tissue_names[train_tissue_indices]
	val_tissue_names = all_tissue_names[val_tissue_indices]

	# Load in gene-based model training/evaluation data
	gene_based_model_data = load_in_gene_based_model_data(prediction_input_data_summary_filestem, min_snps_per_gene=50)
	tot_n_genes = len(gene_based_model_data)
	n_train_val_genes = int(np.floor(tot_n_genes*.8))
	# Split into train/val and test
	train_val_gene_based_model_data = gene_based_model_data[:n_train_val_genes]
	test_gene_based_model_data = gene_based_model_data[n_train_val_genes:]
	
	# Load in gene expression matrices
	gene_expression_data, ge_tissue_names = load_in_expression_data(single_samp_per_tissue_expr_file)
	if np.array_equal(ge_tissue_names, all_tissue_names) == False:
		print('assumption eroror')
		pdb.set_trace()
	
	gene_means = np.mean(gene_expression_data[:, train_tissue_indices], axis=1, keepdims=True)
	gene_stds = np.std(gene_expression_data[:, train_tissue_indices], axis=1, keepdims=True)
	gene_stds[gene_stds == 0.0] = 1.0
	gene_expression_data = (gene_expression_data - gene_means) / gene_stds


	###########################
	# Ready for model training
	############################
	max_epochs=30
	use_held_out_genes_for_validation=True
	train_gene_based_model_data, val_gene_based_model_data = split_train_and_val_gene_based_model_data(train_val_gene_based_model_data, use_held_out_genes_for_validation)
	# Train
	tissue_encoder, variant_encoder = train_model(
		train_gene_based_model_data,
		val_gene_based_model_data,
		gene_expression_data,
		train_tissue_indices,
		val_tissue_indices,
		learning_rate,
		l2_tissue_reg_strength,
		l2_variant_reg_strength,
		variant_encoder_architecture,
		dropout_rate,
		max_epochs=max_epochs,
		use_held_out_genes_for_validation=use_held_out_genes_for_validation
	)
	# Evaluate
	evaluate_model(tissue_encoder, variant_encoder, gene_based_model_data, train_gene_based_model_data, val_gene_based_model_data, test_gene_based_model_data, gene_expression_data, test_tissue_index, model_training_output_stem)
	# Save model results
	tissue_encoder.save(model_training_output_stem + '_tissue_encoder.keras')
	variant_encoder.save(model_training_output_stem + '_variant_encoder.keras')




if __name__ == "__main__":
	main()
