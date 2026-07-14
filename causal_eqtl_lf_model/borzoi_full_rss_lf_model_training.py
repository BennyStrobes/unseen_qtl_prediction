import argparse
import json
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
	print('################################')
	print('Currently subsetting to 1/10th the number of genes for development purposes. NEEED TO CHANGE IN FUTURE')
	print('#################################')

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

			if np.random.choice(np.arange(10)) == 1:
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
	parser.add_argument('--single-samp-per-tissue-expr-file', required=True, type=str)
	parser.add_argument('--prediction-input-data-summary-filestem', required=True, type=str)
	parser.add_argument('--test-tissue', required=True, type=str)
	parser.add_argument('--model-training-output-stem', required=True, type=str)
	parser.add_argument('--n-val-tissues', required=False, type=int, default=5)
	parser.add_argument('--learning-rate', required=False, type=float, default=3e-4)
	parser.add_argument('--l2-tissue-reg-strength', required=False, type=float, default=1e-5)
	parser.add_argument('--l1-variant-reg-strength', required=False, type=float, default=1.0)
	parser.add_argument('--beta-scale', required=False, type=float, default=1e-8,
						help='Scale c on the causal effect beta = c * (U . V^T); constrains prediction magnitude.')
	parser.add_argument('--n-factors', required=False, type=int, default=10)
	return parser.parse_args()


def train_model(gene_based_model_data, gene_expression_data, train_tissue_indices, val_tissue_indices, learning_rate, l2_tissue_reg_strength, l1_variant_reg_strength, n_factors, model_training_output_stem, beta_scale=1e-8, max_epochs=100):
	# Load in number data dimensions
	#n_borzoi_dimensions = np.load(gene_based_model_data[0][6]).shape[1]
	n_expr_gene_dimensions = gene_expression_data.shape[0]

	# Load in borzoi means and standard deviations for each feature
	'''
	borzoi_feature_means, borzoi_feature_sdevs = extract_mean_and_sdev_of_each_borzoi_feature(gene_based_model_data)
	borzoi_feature_means = borzoi_feature_means.astype(np.float32)
	borzoi_feature_sdevs[borzoi_feature_sdevs == 0.0] = 1.0
	borzoi_feature_inv_sdevs = 1.0*(1.0 / borzoi_feature_sdevs).astype(np.float32)
	'''

	n_training_genes = len(gene_based_model_data)
	best_val_loss = np.inf
	best_tissue_weights = None
	best_variant_weights = None
	# clipnorm caps the per-step update so a single high-inv_LD gene can't blow up the weights
	# (the RSS loss has large, ill-conditioned gradients once U is free to move).
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

	# Tissue latent factors V: amortized encoder mapping tissue expression (G-dim) to a K-dim factor.
	l2_reg_tissue = tf.keras.regularizers.l2(l2_tissue_reg_strength)
	tissue_input = tf.keras.Input(shape=(n_expr_gene_dimensions,), name='tissue_expression_input')
	tissue_hidden = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2_reg_tissue, name='tissue_dense_1')(tissue_input)
	tissue_hidden = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2_reg_tissue, name='tissue_dense_2')(tissue_hidden)
	tissue_embedding = tf.keras.layers.Dense(n_factors, activation='linear', kernel_regularizer=l2_reg_tissue, name='tissue_embedding')(tissue_hidden)
	tissue_encoder = tf.keras.Model(inputs=tissue_input, outputs=tissue_embedding, name='tissue_encoder')

	# Variant latent factors U: a free embedding with one K-dim row per (gene, SNP) pair.
	# Each gene is assigned a contiguous block of rows in the global embedding table; the
	# row order within a block matches that gene's LD / z-score / N_eff row order.
	gene_to_u_block = {}
	u_offset = 0
	for gene_data in gene_based_model_data:
		gene_name = gene_data[0]
		gene_n_snps = gene_data[7]
		gene_to_u_block[gene_name] = (u_offset, gene_n_snps)
		u_offset += gene_n_snps
	n_total_variants = u_offset

	# L1 on U is applied manually in train_step, on only the gene's gathered rows (per-gene SGD),
	# rather than via an embeddings_regularizer that would decay the whole table on every step.
	variant_embedding = tf.keras.layers.Embedding(
		input_dim=n_total_variants,
		output_dim=n_factors,
		embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
		name='variant_embedding',
	)
	variant_embedding.build((None,))

	train_tissue_expression_tf = tf.convert_to_tensor(gene_expression_data[:, train_tissue_indices].T.astype(np.float32))
	val_tissue_expression_tf = tf.convert_to_tensor(gene_expression_data[:, val_tissue_indices].T.astype(np.float32))

	# ----------------------------------------------------------------------------------
	# Forward pass + losses (run eagerly: each gene has a different SNP count, so a
	# tf.function would retrace per gene).
	# ----------------------------------------------------------------------------------
	def forward_gene(u_indices_tf, gene_zeds_tf, gene_N_eff_tf, gene_LD_tf, gene_inv_LD_tf, tissue_expression_tf, training):
		# U_g: (n_snps, K) variant factors for this gene; V: (n_tissues, K) tissue factors
		U_g = variant_embedding(u_indices_tf)
		V = tissue_encoder(tissue_expression_tf, training=training)
		# Drop tissues with a missing (NaN) observed z-score BEFORE any multiply, so that
		# NaNs in gene_N_eff / gene_zeds for those tissues never enter the graph. (Masking
		# after the multiply leaves 0 * NaN = NaN in the backward pass -> all-NaN gradients.)
		valid_tissues = ~tf.reduce_any(tf.math.is_nan(gene_zeds_tf), axis=0)
		V = tf.boolean_mask(V, valid_tissues, axis=0)                       # (T_valid, K)
		gene_N_eff_tf = tf.boolean_mask(gene_N_eff_tf, valid_tissues, axis=1)
		obs = tf.boolean_mask(gene_zeds_tf, valid_tissues, axis=1)
		# Standardized causal effects: beta[snp, tissue] = c * (U_g . V^T); c constrains magnitude
		beta_mat = beta_scale * tf.matmul(U_g, V, transpose_b=True)
		# RSS-predicted z-scores (standardized effects: no genotype-sd / AF scaling)
		pred = tf.sqrt(gene_N_eff_tf) * tf.matmul(gene_LD_tf, beta_mat)
		residuals = obs - pred
		# Summed RSS quadratic-form loss: sum over tissues of resid^T inv_LD resid
		loss = tf.reduce_sum(residuals * tf.matmul(gene_inv_LD_tf, residuals))
		return loss, U_g, obs, pred

	def train_step(u_indices_tf, gene_zeds_tf, gene_N_eff_tf, gene_LD_tf, gene_inv_LD_tf, tissue_expression_tf):
		with tf.GradientTape() as tape:
			loss, U_g, _, _ = forward_gene(u_indices_tf, gene_zeds_tf, gene_N_eff_tf, gene_LD_tf, gene_inv_LD_tf, tissue_expression_tf, training=True)
			# Reg: tissue-encoder kernels (whole net, L2) + L1 sparsity on this gene's variant rows
			reg_loss = l1_variant_reg_strength * tf.reduce_sum(tf.abs(U_g))
			if tissue_encoder.losses:
				reg_loss = reg_loss + tf.add_n(tissue_encoder.losses)
			total_loss = loss + reg_loss
		trainable_vars = tissue_encoder.trainable_variables + variant_embedding.trainable_variables
		grads = tape.gradient(total_loss, trainable_vars)
		optimizer.apply_gradients(zip(grads, trainable_vars))
		return loss

	def compute_gene_loss(u_indices_tf, gene_zeds_tf, gene_N_eff_tf, gene_LD_tf, gene_inv_LD_tf, tissue_expression_tf, training=False, compute_correlation=False):
		loss, U_g, obs, pred = forward_gene(u_indices_tf, gene_zeds_tf, gene_N_eff_tf, gene_LD_tf, gene_inv_LD_tf, tissue_expression_tf, training=training)
		if not compute_correlation:
			return loss
		obs_flat = tf.reshape(obs, [-1])
		pred_flat = tf.reshape(pred, [-1])
		if tf.size(obs_flat) > 1:
			obs_c = obs_flat - tf.reduce_mean(obs_flat)
			pred_c = pred_flat - tf.reduce_mean(pred_flat)
			eps = tf.constant(1e-8, dtype=obs_flat.dtype)
			corr = tf.reduce_sum(obs_c * pred_c) / (tf.sqrt(tf.reduce_sum(tf.square(obs_c)) + eps) * tf.sqrt(tf.reduce_sum(tf.square(pred_c)) + eps))
		else:
			corr = tf.constant(np.nan, dtype=loss.dtype)
		return loss, corr


	# Per-epoch training log
	training_log_file = open(model_training_output_stem + '_training_log.txt', 'w')
	training_log_file.write('epoch\ttrain_loss\tval_loss\tval_corr\tbest\n')

	# Epoch training
	for epoch_iter in range(max_epochs):
		print('epoch ' + str(epoch_iter), flush=True)
		epoch_train_losses = []

		t1 = time.time()
		# Loop through genes
		for gene_index in np.random.permutation(n_training_genes):
			# Load in training data for this gene
			gene_name, gene_snp_summary_file, gene_zed_file, gene_N_eff_file, gene_LD_file, gene_inv_LD_file, gene_borzoi_pred_file, n_gene_snps = gene_based_model_data[gene_index]
			gene_LD = np.load(gene_LD_file)
			gene_inv_LD = np.load(gene_inv_LD_file)
			#gene_borzoi_preds = np.load(gene_borzoi_pred_file)
			gene_zeds = np.load(gene_zed_file)
			gene_N_eff = np.load(gene_N_eff_file)
			#gene_afs = np.loadtxt(gene_snp_summary_file,dtype=str)[1:,-1].astype(float)


			# Look up this gene's contiguous block of variant (U) rows
			u_offset, u_n_snps = gene_to_u_block[gene_name]
			if gene_LD.shape[0] != u_n_snps:
				print('assumption error: LD dimension does not match U block size')
				pdb.set_trace()
			if gene_LD.shape[0] != gene_inv_LD.shape[0]:
				print('assumption erroror')
				pdb.set_trace()

			gene_N_eff = gene_N_eff[:, train_tissue_indices]
			gene_zeds = gene_zeds[:, train_tissue_indices]

			u_indices_tf = tf.range(u_offset, u_offset + u_n_snps, dtype=tf.int32)
			gene_LD_tf = tf.convert_to_tensor(gene_LD.astype(np.float32))
			gene_inv_LD_tf = tf.convert_to_tensor(gene_inv_LD.astype(np.float32))
			gene_N_eff_tf = tf.convert_to_tensor(gene_N_eff.astype(np.float32))
			gene_zeds_tf = tf.convert_to_tensor(gene_zeds.astype(np.float32))

			train_loss = train_step(u_indices_tf, gene_zeds_tf, gene_N_eff_tf, gene_LD_tf, gene_inv_LD_tf, train_tissue_expression_tf)
			epoch_train_losses.append(train_loss.numpy())


		epoch_val_losses = []
		epoch_val_corrs = []
		# Validation is on held-out tissues for the SAME genes (U is only learned for training genes).
		for gene_name, gene_snp_summary_file, gene_zed_file, gene_N_eff_file, gene_LD_file, gene_inv_LD_file, gene_borzoi_pred_file, n_gene_snps in gene_based_model_data:
			gene_LD = np.load(gene_LD_file)
			gene_inv_LD = np.load(gene_inv_LD_file)
			gene_zeds = np.load(gene_zed_file)
			gene_N_eff = np.load(gene_N_eff_file)

			u_offset, u_n_snps = gene_to_u_block[gene_name]
			gene_N_eff = gene_N_eff[:, val_tissue_indices]
			gene_zeds = gene_zeds[:, val_tissue_indices]

			# Skip genes with no valid val tissue: an empty tissue set makes the summed RSS
			# loss 0.0 (not NaN), which would slip past the NaN filter and deflate val_loss.
			valid_val_tissues = ~np.isnan(gene_zeds).any(axis=0)
			if valid_val_tissues.sum() == 0:
				continue

			u_indices_tf = tf.range(u_offset, u_offset + u_n_snps, dtype=tf.int32)
			gene_LD_tf = tf.convert_to_tensor(gene_LD.astype(np.float32))
			gene_inv_LD_tf = tf.convert_to_tensor(gene_inv_LD.astype(np.float32))
			gene_N_eff_tf = tf.convert_to_tensor(gene_N_eff.astype(np.float32))
			gene_zeds_tf = tf.convert_to_tensor(gene_zeds.astype(np.float32))
			val_loss, val_corr = compute_gene_loss(
				u_indices_tf,
				gene_zeds_tf,
				gene_N_eff_tf,
				gene_LD_tf,
				gene_inv_LD_tf,
				val_tissue_expression_tf,
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
		t2 = time.time()
		if epoch_val_loss < best_val_loss:
			best_val_loss = epoch_val_loss
			best_tissue_weights = tissue_encoder.get_weights()
			best_variant_weights = variant_embedding.get_weights()
			status = ' best'
		print((t2-t1)/60.0, 'minutes', flush=True)
		print('epoch ' + str(epoch_iter) + ' train_loss=' + str(np.mean(epoch_train_losses)) + ' val_loss=' + str(epoch_val_loss) + ' val_corr=' + str(np.mean(epoch_val_corrs)) + status, flush=True)
		training_log_file.write(str(epoch_iter) + '\t' + str(np.mean(epoch_train_losses)) + '\t' + str(epoch_val_loss) + '\t' + str(np.mean(epoch_val_corrs)) + '\t' + str(status == ' best') + '\n')
		training_log_file.flush()

	training_log_file.close()

	if best_tissue_weights is not None:
		tissue_encoder.set_weights(best_tissue_weights)
		variant_embedding.set_weights(best_variant_weights)

	return tissue_encoder, variant_embedding, gene_to_u_block


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
	l1_variant_reg_strength = args.l1_variant_reg_strength
	beta_scale = args.beta_scale
	n_factors = args.n_factors

	np.random.seed(1)

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
	test_tissue_name = all_tissue_names[test_tissue_index]

	# Load in gene-based model training/evaluation data
	gene_based_model_data = load_in_gene_based_model_data(prediction_input_data_summary_filestem, min_snps_per_gene=50)
	tot_n_genes = len(gene_based_model_data)
	
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
	max_epochs=150

	# Train
	tissue_encoder, variant_embedding, gene_to_u_block = train_model(
		gene_based_model_data,
		gene_expression_data,
		train_tissue_indices,
		val_tissue_indices,
		learning_rate,
		l2_tissue_reg_strength,
		l1_variant_reg_strength,
		n_factors,
		model_training_output_stem,
		beta_scale=beta_scale,
		max_epochs=max_epochs
	)

	# Held-out test-tissue evaluation of the single best model across the hyperparameter sweep
	# is a separate downstream step (evaluate_lf_model_on_test_tissue.py, run from driver_key.sh),
	# so this training run just persists its best-epoch model + eval metadata below.

	# Save model results
	# Tissue encoder is a Model -> native .keras save.
	tissue_encoder.save(model_training_output_stem + '_tissue_encoder.keras')
	# variant_embedding is a bare Embedding layer (no .save); persist its (n_total_variants, K)
	# weight matrix, plus the gene -> (row_offset, n_snps) map needed to interpret its rows.
	np.save(model_training_output_stem + '_variant_embedding.npy', variant_embedding.get_weights()[0])
	with open(model_training_output_stem + '_gene_to_u_block.json', 'w') as f:
		json.dump({gene: list(block) for gene, block in gene_to_u_block.items()}, f)
	# Metadata for the downstream held-out test-tissue evaluation. To recompute the test
	# tissue's factor V_test exactly, the evaluator must standardize its expression with the
	# SAME per-gene mean/std that training derived from the TRAIN tissues (these depend on the
	# random tissue split, so they can't be re-derived from the data alone). Also persist the
	# test tissue column index and the causal-effect scale c (beta = c * U . V^T).
	np.savez(
		model_training_output_stem + '_eval_metadata.npz',
		gene_expression_means=gene_means[:, 0],
		gene_expression_stds=gene_stds[:, 0],
		test_tissue_index=test_tissue_index,
		train_tissue_indices=train_tissue_indices,
		val_tissue_indices=val_tissue_indices,
		beta_scale=beta_scale,
	)




if __name__ == "__main__":
	main()
