import argparse
import json
import numpy as np
import os
import pdb
import tensorflow as tf
import time


# ======================================================================================
# Closed-form-U variant of borzoi_full_rss_lf_model_training.py
#
# Same model (beta[snp, tissue] = U_g . V^T, RSS quadratic-form loss on z-scores, amortized
# tissue encoder V = f(expression)) and same data / tissue split / gene subset / outputs, but
# the variant factors U are no longer learned by one Adam step per gene per epoch. Instead,
# every time a gene is visited, U_g is set to its exact minimizer given the current V
# (block coordinate descent), and only the tissue encoder takes a gradient step.
#
# Derivation. For one gene with LD R, inverse LD Lambda, per-tissue z-scores z_t and sample
# size N_t, the loss is sum_t r_t^T Lambda r_t with r_t = z_t - sqrt(N_t) R U v_t. Setting
# d/dU = 0 gives
#     R Lambda R U G = R Lambda S,   G = sum_t N_t v_t v_t^T (K x K),   S = sum_t sqrt(N_t) z_t v_t^T (M x K)
# When Lambda = R^{-1} this collapses to  U = Lambda S G^{-1}:  an M x M matrix we already load,
# times an M x K matrix, times a K x K inverse. The LD weighting cancels because it is shared
# across tissues (U is the least-squares fit of the per-tissue joint effects Lambda z_t / sqrt(N_t)
# onto the factor model). If Lambda is a regularized inverse this is approximate in the
# low-eigenvalue directions of R; a one-time diagnostic (||R Lambda R - R|| / ||R||) is printed
# for the first few genes so that approximation can be judged.
#
# Two consequences handled below:
#  * U V^T is invariant to U -> aU, V -> V/a, so an exact U given V would absorb any scale the
#    encoder drifts to, and the L1 on U would be meaningless. V is therefore normalized to unit
#    RMS over the training tissues (a single scalar, so relative tissue magnitudes are kept) and
#    U is in standardized-effect units. The scalar is folded into the saved encoder so the
#    downstream evaluator (evaluate_lf_model_on_test_tissue.py) needs no change.
#  * The L1 on U is applied as a coordinate-wise soft-threshold after the solve (one proximal
#    step from the least-squares solution; approximate lasso, ignores LD coupling between SNPs).
#    The threshold is l1 / (2 * diag(G)), i.e. scaled by the per-coordinate curvature, so with
#    V at unit RMS the strengths that actually bite are roughly 1 to 100.
# ======================================================================================


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
		f.close()

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


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gtex-tissue-names-file', required=True, type=str)
	parser.add_argument('--single-samp-per-tissue-expr-file', required=True, type=str)
	parser.add_argument('--prediction-input-data-summary-filestem', required=True, type=str)
	parser.add_argument('--test-tissue', required=True, type=str)
	parser.add_argument('--model-training-output-stem', required=True, type=str)
	parser.add_argument('--n-val-tissues', required=False, type=int, default=5)
	parser.add_argument('--learning-rate', required=False, type=float, default=3e-4,
						help='Adam learning rate for the tissue encoder (U has no learning rate: it is solved in closed form).')
	parser.add_argument('--l2-tissue-reg-strength', required=False, type=float, default=1e-5)
	parser.add_argument('--l1-variant-reg-strength', required=False, type=float, default=1.0,
						help='Soft-threshold (approximate lasso) strength on U after each closed-form solve. With V at unit RMS, values ~1-100 are where it bites.')
	parser.add_argument('--u-ridge-frac', required=False, type=float, default=1e-2,
						help='Ridge added to the K x K Gram matrix in the closed-form U solve, as a fraction of its mean diagonal.')
	parser.add_argument('--encoder-genes-per-step', required=False, type=int, default=32,
						help='Number of genes whose encoder gradients are averaged before each Adam step on the encoder. 1 = one step per gene (U is still solved per gene regardless).')
	parser.add_argument('--beta-scale', required=False, type=float, default=1.0,
						help='Scale c on the causal effect beta = c * (U . V^T). Kept at 1.0 here; U is solved in effect-size units.')
	parser.add_argument('--n-factors', required=False, type=int, default=10)
	return parser.parse_args()


def closed_form_u_update(gene_inv_LD, gene_zeds, gene_N_eff, V, n_factors, u_ridge_frac, l1_variant_reg_strength):
	# Exact minimizer of this gene's RSS loss over U_g given V (see header), in float64.
	#   gene_inv_LD: (M, M)   gene_zeds: (M, T) with all-NaN columns for unobserved tissues
	#   gene_N_eff:  (M, T)   V: (T, K) normalized tissue factors (training tissues only)
	# Returns (M, K) float32 U_g, or None if no tissue is observed for this gene.
	valid_tissues = ~np.isnan(gene_zeds).any(axis=0)
	if valid_tissues.sum() == 0:
		return None
	Z = gene_zeds[:, valid_tissues].astype(np.float64)                       # (M, T_valid)
	# N_eff is stored per SNP; the closed form needs one N per tissue, so use the per-tissue
	# mean over SNPs (N_eff varies little across SNPs within a tissue).
	N_t = np.nanmean(gene_N_eff[:, valid_tissues].astype(np.float64), axis=0)   # (T_valid,)
	sqrt_N_t = np.sqrt(N_t)
	V_valid = V[valid_tissues, :].astype(np.float64)                            # (T_valid, K)

	# S = Lambda @ sum_t sqrt(N_t) z_t v_t^T  ->  (M, K)
	S = (gene_inv_LD.astype(np.float64) @ Z) @ (sqrt_N_t[:, None] * V_valid)
	# G = sum_t N_t v_t v_t^T  ->  (K, K), plus a relative ridge for conditioning
	G = (V_valid * N_t[:, None]).T @ V_valid
	ridge = u_ridge_frac * np.trace(G) / float(n_factors)
	G_ridged = G + ridge * np.eye(n_factors)
	# U G = S  ->  U = S G^{-1}; solve the transposed system (G is symmetric)
	U_g = np.linalg.solve(G_ridged, S.T).T                                      # (M, K)

	# Approximate lasso: one coordinate-wise proximal step from the least-squares solution.
	# Per-coordinate curvature of the loss in U_ik is 2 * G_kk (diag(R) = 1), so the threshold
	# on coordinate k is l1 / (2 * G_kk).
	if l1_variant_reg_strength > 0.0:
		thresholds = l1_variant_reg_strength / (2.0 * np.diag(G_ridged))         # (K,)
		U_g = np.sign(U_g) * np.maximum(np.abs(U_g) - thresholds[None, :], 0.0)

	return U_g.astype(np.float32)


def train_model(gene_based_model_data, gene_expression_data, train_tissue_indices, val_tissue_indices, learning_rate, l2_tissue_reg_strength, l1_variant_reg_strength, u_ridge_frac, n_factors, model_training_output_stem, beta_scale=1.0, max_epochs=100, encoder_genes_per_step=32):
	n_expr_gene_dimensions = gene_expression_data.shape[0]
	n_training_genes = len(gene_based_model_data)
	best_val_loss = np.inf
	best_tissue_weights = None
	best_variant_table = None
	# Adam is now only for the tissue encoder. clipnorm kept from the original trainer.
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

	# Tissue latent factors V: amortized encoder mapping tissue expression (G-dim) to a K-dim factor.
	l2_reg_tissue = tf.keras.regularizers.l2(l2_tissue_reg_strength)
	tissue_input = tf.keras.Input(shape=(n_expr_gene_dimensions,), name='tissue_expression_input')
	tissue_hidden = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2_reg_tissue, name='tissue_dense_1')(tissue_input)
	tissue_hidden = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2_reg_tissue, name='tissue_dense_2')(tissue_hidden)
	tissue_embedding = tf.keras.layers.Dense(n_factors, activation='linear', kernel_regularizer=l2_reg_tissue, name='tissue_embedding')(tissue_hidden)
	tissue_encoder = tf.keras.Model(inputs=tissue_input, outputs=tissue_embedding, name='tissue_encoder')

	# Variant latent factors U: one K-dim row per (gene, SNP) pair, held in a plain numpy table
	# (no gradients flow to it). Each gene owns a contiguous block of rows; row order within a
	# block matches that gene's LD / z-score / N_eff row order. Initialized to zero: every block
	# is overwritten by its closed-form solve on the gene's first visit, before it is ever used.
	gene_to_u_block = {}
	u_offset = 0
	for gene_data in gene_based_model_data:
		gene_name = gene_data[0]
		gene_n_snps = gene_data[7]
		gene_to_u_block[gene_name] = (u_offset, gene_n_snps)
		u_offset += gene_n_snps
	n_total_variants = u_offset
	variant_table = np.zeros((n_total_variants, n_factors), dtype=np.float32)

	train_tissue_expression_tf = tf.convert_to_tensor(gene_expression_data[:, train_tissue_indices].T.astype(np.float32))
	val_tissue_expression_tf = tf.convert_to_tensor(gene_expression_data[:, val_tissue_indices].T.astype(np.float32))

	# ----------------------------------------------------------------------------------
	# Encoder with the scale pinned: V = f(expr) / s, where s is the RMS of f over the TRAINING
	# tissues. Removes the U <-> V scale ambiguity so U stays in effect-size units.
	# ----------------------------------------------------------------------------------
	def train_tissue_scale(training):
		V_train_raw = tissue_encoder(train_tissue_expression_tf, training=training)
		return tf.sqrt(tf.reduce_mean(tf.square(V_train_raw)) + 1e-12)

	def encode(tissue_expression_tf, training):
		s = train_tissue_scale(training)
		return tissue_encoder(tissue_expression_tf, training=training) / s

	# ----------------------------------------------------------------------------------
	# Forward pass + losses (run eagerly: each gene has a different SNP count). U_g enters as a
	# constant tensor, so gradients reach only the encoder.
	# ----------------------------------------------------------------------------------
	def forward_gene(U_g_tf, gene_zeds_tf, gene_N_eff_tf, gene_LD_tf, gene_inv_LD_tf, tissue_expression_tf, training):
		V = encode(tissue_expression_tf, training)
		# Drop tissues with a missing (NaN) observed z-score BEFORE any multiply (see original trainer).
		valid_tissues = ~tf.reduce_any(tf.math.is_nan(gene_zeds_tf), axis=0)
		V = tf.boolean_mask(V, valid_tissues, axis=0)                       # (T_valid, K)
		gene_N_eff_tf = tf.boolean_mask(gene_N_eff_tf, valid_tissues, axis=1)
		obs = tf.boolean_mask(gene_zeds_tf, valid_tissues, axis=1)
		beta_mat = beta_scale * tf.matmul(U_g_tf, V, transpose_b=True)
		pred = tf.sqrt(gene_N_eff_tf) * tf.matmul(gene_LD_tf, beta_mat)
		residuals = obs - pred
		loss = tf.reduce_sum(residuals * tf.matmul(gene_inv_LD_tf, residuals))
		return loss, obs, pred

	trainable_vars = tissue_encoder.trainable_variables

	def encoder_gradients(U_g_tf, gene_zeds_tf, gene_N_eff_tf, gene_LD_tf, gene_inv_LD_tf, tissue_expression_tf):
		# Loss + encoder gradients for one gene (U_g fixed). The encoder L2 is included per gene, so
		# averaging over a block of genes keeps it at the same relative strength as a 1-gene step.
		with tf.GradientTape() as tape:
			loss, _, _ = forward_gene(U_g_tf, gene_zeds_tf, gene_N_eff_tf, gene_LD_tf, gene_inv_LD_tf, tissue_expression_tf, training=True)
			total_loss = loss
			if tissue_encoder.losses:
				total_loss = total_loss + tf.add_n(tissue_encoder.losses)
		grads = tape.gradient(total_loss, trainable_vars)
		return loss, grads

	# Gradient accumulation over encoder_genes_per_step genes: gradients are summed per gene and
	# the MEAN is applied in one Adam step (mean rather than sum so clipnorm behaves the same as a
	# 1-gene step). With encoder_genes_per_step == 1 this is exactly one step per gene.
	accumulated_grads = [tf.zeros_like(v) for v in trainable_vars]
	accumulated_count = 0

	def accumulate_and_maybe_step(grads, force=False):
		nonlocal accumulated_grads, accumulated_count
		if grads is not None:
			accumulated_grads = [a + g for a, g in zip(accumulated_grads, grads)]
			accumulated_count += 1
		if accumulated_count == 0:
			return
		if accumulated_count >= encoder_genes_per_step or force:
			mean_grads = [a / float(accumulated_count) for a in accumulated_grads]
			optimizer.apply_gradients(zip(mean_grads, trainable_vars))
			accumulated_grads = [tf.zeros_like(v) for v in trainable_vars]
			accumulated_count = 0

	def compute_gene_loss(U_g_tf, gene_zeds_tf, gene_N_eff_tf, gene_LD_tf, gene_inv_LD_tf, tissue_expression_tf, compute_correlation=False):
		loss, obs, pred = forward_gene(U_g_tf, gene_zeds_tf, gene_N_eff_tf, gene_LD_tf, gene_inv_LD_tf, tissue_expression_tf, training=False)
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

	# One-time diagnostic: how far is the stored inverse LD from a true inverse? The closed form
	# is exact when R Lambda R == R.
	for gene_data in gene_based_model_data[:3]:
		R = np.load(gene_data[4]).astype(np.float64)
		Lam = np.load(gene_data[5]).astype(np.float64)
		rel_err = np.linalg.norm(R @ Lam @ R - R) / np.linalg.norm(R)
		print('closed-form diagnostic ' + gene_data[0] + ': ||R Lambda R - R||_F / ||R||_F = ' + str(rel_err), flush=True)

	# Per-epoch training log (same format as the original trainer)
	training_log_file = open(model_training_output_stem + '_training_log.txt', 'w')
	training_log_file.write('epoch\ttrain_loss\tval_loss\tval_corr\tbest\n')

	for epoch_iter in range(max_epochs):
		print('epoch ' + str(epoch_iter), flush=True)
		epoch_train_losses = []

		t1 = time.time()
		for gene_index in np.random.permutation(n_training_genes):
			gene_name, gene_snp_summary_file, gene_zed_file, gene_N_eff_file, gene_LD_file, gene_inv_LD_file, gene_borzoi_pred_file, n_gene_snps = gene_based_model_data[gene_index]
			gene_LD = np.load(gene_LD_file)
			gene_inv_LD = np.load(gene_inv_LD_file)
			gene_zeds = np.load(gene_zed_file)
			gene_N_eff = np.load(gene_N_eff_file)

			u_offset, u_n_snps = gene_to_u_block[gene_name]
			if gene_LD.shape[0] != u_n_snps:
				print('assumption error: LD dimension does not match U block size')
				pdb.set_trace()
			if gene_LD.shape[0] != gene_inv_LD.shape[0]:
				print('assumption erroror')
				pdb.set_trace()

			gene_N_eff = gene_N_eff[:, train_tissue_indices]
			gene_zeds = gene_zeds[:, train_tissue_indices]

			# (1) Closed-form U_g given the current (normalized) training-tissue factors V
			V_train = encode(train_tissue_expression_tf, training=False).numpy()
			U_g = closed_form_u_update(gene_inv_LD, gene_zeds, gene_N_eff, V_train, n_factors, u_ridge_frac, l1_variant_reg_strength)
			if U_g is None:
				continue
			variant_table[u_offset:u_offset + u_n_snps, :] = U_g

			# (2) Encoder gradients with U_g held fixed; one Adam step per encoder_genes_per_step genes
			U_g_tf = tf.convert_to_tensor(U_g)
			gene_LD_tf = tf.convert_to_tensor(gene_LD.astype(np.float32))
			gene_inv_LD_tf = tf.convert_to_tensor(gene_inv_LD.astype(np.float32))
			gene_N_eff_tf = tf.convert_to_tensor(gene_N_eff.astype(np.float32))
			gene_zeds_tf = tf.convert_to_tensor(gene_zeds.astype(np.float32))

			train_loss, grads = encoder_gradients(U_g_tf, gene_zeds_tf, gene_N_eff_tf, gene_LD_tf, gene_inv_LD_tf, train_tissue_expression_tf)
			accumulate_and_maybe_step(grads)
			epoch_train_losses.append(train_loss.numpy())

		# Flush a partial block at the end of the epoch so no gene's gradient is dropped
		accumulate_and_maybe_step(None, force=True)

		epoch_val_losses = []
		epoch_val_corrs = []
		# Validation is on held-out tissues for the SAME genes (U is only solved for training genes).
		for gene_name, gene_snp_summary_file, gene_zed_file, gene_N_eff_file, gene_LD_file, gene_inv_LD_file, gene_borzoi_pred_file, n_gene_snps in gene_based_model_data:
			gene_LD = np.load(gene_LD_file)
			gene_inv_LD = np.load(gene_inv_LD_file)
			gene_zeds = np.load(gene_zed_file)
			gene_N_eff = np.load(gene_N_eff_file)

			u_offset, u_n_snps = gene_to_u_block[gene_name]
			gene_N_eff = gene_N_eff[:, val_tissue_indices]
			gene_zeds = gene_zeds[:, val_tissue_indices]

			valid_val_tissues = ~np.isnan(gene_zeds).any(axis=0)
			if valid_val_tissues.sum() == 0:
				continue

			U_g_tf = tf.convert_to_tensor(variant_table[u_offset:u_offset + u_n_snps, :])
			gene_LD_tf = tf.convert_to_tensor(gene_LD.astype(np.float32))
			gene_inv_LD_tf = tf.convert_to_tensor(gene_inv_LD.astype(np.float32))
			gene_N_eff_tf = tf.convert_to_tensor(gene_N_eff.astype(np.float32))
			gene_zeds_tf = tf.convert_to_tensor(gene_zeds.astype(np.float32))
			val_loss, val_corr = compute_gene_loss(
				U_g_tf,
				gene_zeds_tf,
				gene_N_eff_tf,
				gene_LD_tf,
				gene_inv_LD_tf,
				val_tissue_expression_tf,
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
			best_variant_table = variant_table.copy()
			status = ' best'
		print((t2-t1)/60.0, 'minutes', flush=True)
		print('epoch ' + str(epoch_iter) + ' train_loss=' + str(np.mean(epoch_train_losses)) + ' val_loss=' + str(epoch_val_loss) + ' val_corr=' + str(np.mean(epoch_val_corrs)) + status, flush=True)
		training_log_file.write(str(epoch_iter) + '\t' + str(np.mean(epoch_train_losses)) + '\t' + str(epoch_val_loss) + '\t' + str(np.mean(epoch_val_corrs)) + '\t' + str(status == ' best') + '\n')
		training_log_file.flush()

	training_log_file.close()

	if best_tissue_weights is not None:
		tissue_encoder.set_weights(best_tissue_weights)
		variant_table = best_variant_table

	# Fold the training-tissue RMS scale into the final (linear) layer so that the saved encoder
	# returns the normalized V directly. The evaluator calls the saved encoder as-is.
	s = train_tissue_scale(training=False).numpy()
	embedding_layer = tissue_encoder.get_layer('tissue_embedding')
	kernel, bias = embedding_layer.get_weights()
	embedding_layer.set_weights([kernel / s, bias / s])

	return tissue_encoder, variant_table, gene_to_u_block


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
	u_ridge_frac = args.u_ridge_frac
	encoder_genes_per_step = args.encoder_genes_per_step
	beta_scale = args.beta_scale
	n_factors = args.n_factors
	if encoder_genes_per_step < 1:
		raise ValueError('--encoder-genes-per-step must be >= 1')

	# Same seeds and same draw order as the original trainer, so the validation tissues and the
	# gene subset are identical for a given test tissue.
	np.random.seed(1)
	tf.random.set_seed(1)

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
	# Scale the standardized expression down by the number of genes (Adam first-layer step size;
	# see the original trainer). Folded into gene_stds so the saved eval metadata applies it too.
	n_expression_genes = gene_expression_data.shape[0]
	gene_stds = gene_stds * n_expression_genes
	gene_expression_data = (gene_expression_data - gene_means) / gene_stds

	###########################
	# Ready for model training
	############################
	max_epochs=200

	tissue_encoder, variant_table, gene_to_u_block = train_model(
		gene_based_model_data,
		gene_expression_data,
		train_tissue_indices,
		val_tissue_indices,
		learning_rate,
		l2_tissue_reg_strength,
		l1_variant_reg_strength,
		u_ridge_frac,
		n_factors,
		model_training_output_stem,
		beta_scale=beta_scale,
		max_epochs=max_epochs,
		encoder_genes_per_step=encoder_genes_per_step
	)

	# Save model results in the same layout as the original trainer so
	# summarize_lf_model_training_logs.py / evaluate_lf_model_on_test_tissue.py work unchanged.
	tissue_encoder.save(model_training_output_stem + '_tissue_encoder.keras')
	np.save(model_training_output_stem + '_variant_embedding.npy', variant_table)
	with open(model_training_output_stem + '_gene_to_u_block.json', 'w') as f:
		json.dump({gene: list(block) for gene, block in gene_to_u_block.items()}, f)
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
