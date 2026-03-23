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


# ============================================================
# 1. MODEL DEFINITIONS
# ============================================================

class TissueNet(tf.keras.Model):
	"""
	Maps tissue features x_t (G-dim) to:
	  - latent factor v_t (K-dim)
	  - tissue-specific residual variance sigma_t^2 (scalar, >= 0)
	"""
	def __init__(self, G, K, hidden=64, dropout_rate=0.0, **kwargs):
		super().__init__(**kwargs)
		self.dense1 = layers.Dense(hidden, activation="relu")
		self.dense2 = layers.Dense(hidden, activation="relu")
		self.dense3 = layers.Dense(hidden, activation="relu")
		self.dropout = layers.Dropout(dropout_rate)

		# Mean head
		self.out_mean = layers.Dense(K, activation=None)

		# Variance head (log-scale, mapped to positive)
		self.out_logvar = layers.Dense(1, activation=None)

	def call(self, X_t, training=False):
		h = self.dense1(X_t)
		h = self.dense2(h)
		h = self.dense3(h)
		if training:
			h = self.dropout(h, training=training)

		V_T = self.out_mean(h)                     # (T, K)
		logvar_T = self.out_logvar(h)              # (T, 1)
		var_T = tf.nn.softplus(tf.squeeze(logvar_T, axis=1)) + 1e-8  # (T,)

		return V_T, var_T


class EQTLFactorModel(tf.keras.Model):
	def __init__(self, N, G, K, X_train_t,
				 hidden=64, dropout_rate=0.0, **kwargs):
		super().__init__(**kwargs)
		self.N = N
		self.G = G
		self.K = K

		self.U_embed = layers.Embedding(
			input_dim=N,
			output_dim=K,
			embeddings_initializer="glorot_uniform",
			name="U_embedding",
		)

		self.tissue_net = TissueNet(
			G, K,
			hidden=hidden,
			dropout_rate=dropout_rate
		)

		self.X_train_t = tf.constant(X_train_t, dtype=tf.float32)

	def call(self, n_idx, training=False):
		U_batch = self.U_embed(n_idx)                         # (B, K)
		V_train, var_train_t = self.tissue_net(
			self.X_train_t, training=training
		)                                                     # (T_train,K), (T_train,)

		mu_train = tf.einsum("bk,tk->bt", U_batch, V_train)   # (B, T_train)
		return mu_train, var_train_t


# ============================================================
# 2. TRAINER CLASS (train_step, val_step, fit)
# ============================================================

class EQTLTrainer:
	def __init__(
		self,
		model,
		beta_train_tf, se_train_tf, mask_train_tf,
		beta_val_tf, se_val_tf, mask_val_tf,
		X_val_t_tf,
		batch_size=2048,
		learning_rate=1e-3,
	):
		"""
		model:          EQTLFactorModel
		beta_train_tf:  (N, T_train) float32
		se_train_tf:    (N, T_train) float32
		mask_train_tf:  (N, T_train) bool
		beta_val_tf:    (N, T_val)   float32
		se_val_tf:      (N, T_val)   float32
		mask_val_tf:    (N, T_val)   bool
		X_val_t_tf:     (T_val, G)   float32
		"""
		self.model = model
		self.optimizer = tf.keras.optimizers.Adam(learning_rate)
		self.beta_train_tf = beta_train_tf
		self.se_train_tf = se_train_tf
		self.mask_train_tf = mask_train_tf
		self.beta_val_tf = beta_val_tf
		self.se_val_tf = se_val_tf
		self.mask_val_tf = mask_val_tf
		self.X_val_t_tf = X_val_t_tf
		self.batch_size = batch_size
		self.eps = 1e-8

	@tf.function
	def train_step(self, n_idx_batch):
		model = self.model

		with tf.GradientTape() as tape:
			mu_train, var_train_t = model(n_idx_batch, training=True)

			beta_batch = tf.gather(self.beta_train_tf, n_idx_batch)
			se_batch   = tf.gather(self.se_train_tf,   n_idx_batch)
			mask_batch = tf.gather(self.mask_train_tf, n_idx_batch)

			mask_f = tf.cast(mask_batch, tf.float32)

			var_total = tf.square(se_batch) + tf.reshape(var_train_t, (1, -1))
			resid = beta_batch - mu_train

			per_cell = (tf.square(resid) / var_total) + tf.math.log(var_total)
			per_cell = per_cell * mask_f

			denom = tf.reduce_sum(mask_f)
			denom = tf.maximum(denom, self.eps)

			loss = tf.reduce_sum(per_cell) / denom

		grads = tape.gradient(loss, model.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
		return loss

	@tf.function
	def val_step(self, n_idx_batch):
		model = self.model

		U_batch = model.U_embed(n_idx_batch, training=False)
		V_val, var_val_t = model.tissue_net(self.X_val_t_tf, training=False)

		mu_val = tf.einsum("bk,tk->bt", U_batch, V_val)

		beta_val_batch = tf.gather(self.beta_val_tf, n_idx_batch)
		se_val_batch   = tf.gather(self.se_val_tf,   n_idx_batch)
		mask_val_batch = tf.gather(self.mask_val_tf, n_idx_batch)

		mask_f = tf.cast(mask_val_batch, tf.float32)

		var_total = tf.square(se_val_batch) + tf.reshape(var_val_t, (1, -1))
		resid = beta_val_batch - mu_val

		per_cell = (tf.square(resid) / var_total) + tf.math.log(var_total)
		per_cell = per_cell * mask_f

		denom = tf.reduce_sum(mask_f)
		denom = tf.maximum(denom, self.eps)

		loss = tf.reduce_sum(per_cell) / denom
		return loss

	def fit(self, train_ds, val_ds, num_epochs=5):
		best_val_loss2 = np.inf
		best_val_corr_at_best = -np.inf
		best_epoch = -1
		best_weights = None
		status = ''

		for epoch in range(num_epochs):

			# ---- Training ----
			for n_idx_batch in train_ds:
				self.train_step(n_idx_batch)

			# ---- Validation forward pass ----
			val_preds, val_varz = predict_new_tissue(self.model, self.X_val_t_tf)

			val_corr = np.corrcoef(
				self.beta_val_tf[self.mask_val_tf],
				val_preds[self.mask_val_tf]
			)[0, 1]

			val_var_total = (self.se_val_tf.numpy() ** 2) + val_varz[None, :]
			val_resid = self.beta_val_tf.numpy() - val_preds
			val_mask = self.mask_val_tf.numpy().astype(bool)
			val_nll_cells = (val_resid**2) / val_var_total + np.log(val_var_total)
			val_loss2 = np.mean(val_nll_cells[val_mask])

			# ---- SELECT BEST BY LOWEST VAL LOSS ----
			if val_loss2 < best_val_loss2:
				best_val_loss2 = val_loss2
				best_val_corr_at_best = val_corr
				best_epoch = epoch
				best_weights = self.model.get_weights()
				status = 'Best'

			print(
				f"Epoch {epoch+1}/{num_epochs} "
				f"- val_loss: {val_loss2:.5f} "
				f"- val_corr: {val_corr:.5f} "
				f"- {status}"
			)
			status = ''

		if best_weights is not None:
			self.model.set_weights(best_weights)

		return self.model, (best_epoch + 1), best_val_loss2, best_val_corr_at_best


# ============================================================
# 3. UTILS: PREDICT FOR NEW TISSUE
# ============================================================

def predict_new_tissue(model, x_new, n_indices=None):
	if n_indices is None:
		n_indices = np.arange(model.N, dtype=np.int32)
	else:
		n_indices = np.asarray(n_indices, dtype=np.int32)

	n_idx_tf = tf.convert_to_tensor(n_indices, dtype=tf.int32)
	U_batch = model.U_embed(n_idx_tf)   # (B, K)

	V_new, var_new_t = model.tissue_net(x_new, training=False)  # (T, K), (T,)

	pred = tf.matmul(U_batch, V_new, transpose_b=True)         # (B, T)

	return pred.numpy(), var_new_t.numpy()


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
	blocks = np.array_split(arr, n_blocks, axis=0)

	bs_corrz = []
	for bs_iter in range(n_boots):
		bs_blocks = np.random.choice(np.arange(n_blocks), size=n_blocks, replace=True)
		boot_arr = np.concatenate([blocks[i] for i in bs_blocks])
		bs_corrz.append(np.corrcoef(boot_arr[:, 0], boot_arr[:, 1])[0, 1])
	bs_corrz = np.asarray(bs_corrz)
	return np.mean(bs_corrz), np.std(bs_corrz)


def get_test_losses(trained_model, beta_test, se_test, mask_test, X_test_t):
	# Make model test predictions
	model_preds, model_pred_vars = predict_new_tissue(trained_model, X_test_t)

	# Compute test loss
	test_losses = []
	test_losses_ses = []
	test_corrz = []
	test_corrz_ses = []
	tot = []
	for col_iter in range(beta_test.shape[1]):
		# Subset to just this column
		tmp_model_preds = model_preds[:, col_iter]
		tmp_model_pred_vars = model_pred_vars[col_iter]

		tmp_beta_test = beta_test[:, col_iter]
		tmp_se_test = se_test[:, col_iter]
		tmp_mask_test = mask_test[:, col_iter]

		############
		# Get loss
		squared_resids = np.square(tmp_model_preds[tmp_mask_test] - tmp_beta_test[tmp_mask_test])
		denom = np.square(tmp_se_test[tmp_mask_test])
		per_snp_loss = squared_resids / denom
		avg_loss = np.mean(per_snp_loss)
		avg_loss_bs, avg_loss_bs_se = genomic_block_bootstrap_avg(per_snp_loss, n_blocks=100, n_boots=1000)
		# Append to array
		test_losses.append(avg_loss)
		test_losses_ses.append(avg_loss_bs_se)

		tot.append(per_snp_loss)

		############
		# Get correlation
		avg_corr = np.corrcoef(tmp_model_preds[tmp_mask_test], tmp_beta_test[tmp_mask_test])[0, 1]
		avg_corr_bs, avg_corr_bs_se = genomic_block_bootstrap_corr(
			tmp_model_preds[tmp_mask_test],
			tmp_beta_test[tmp_mask_test],
			n_blocks=100,
			n_boots=1000
		)
		test_corrz.append(avg_corr)
		test_corrz_ses.append(avg_corr_bs_se)

	aa = tmp_model_preds[tmp_mask_test]
	bb = tmp_beta_test[tmp_mask_test]
	cc = tmp_se_test[tmp_mask_test]
	dd = tmp_model_pred_vars

	test_losses = np.asarray(test_losses)
	test_losses_ses = np.asarray(test_losses_ses)
	test_corrz = np.asarray(test_corrz)
	test_corrz_ses = np.asarray(test_corrz_ses)

	return test_losses, test_losses_ses, test_corrz, test_corrz_ses, aa, bb, cc, dd


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

	val_tissue_idx = np.sort(np.random.choice(non_test_idx, size=n_validation_tissues, replace=False))

	train_tissue_idx = []
	for tissue_idx in non_test_idx:
		if tissue_idx not in val_tissue_idx:
			train_tissue_idx.append(tissue_idx)
	train_tissue_idx = np.asarray(train_tissue_idx)

	return train_tissue_idx, val_tissue_idx, test_idx


def get_distances(X_train_t, X_val_t):
	n_tiss = X_val_t.shape[0]

	min_distances = []
	avg_distances = []
	for tiss_iter in range(n_tiss):
		distances = np.sum(np.square(X_train_t - X_val_t[tiss_iter,:]),axis=1)

		min_distances.append(np.min(distances))
		avg_distances.append(np.mean(distances))

	return np.asarray(min_distances), np.asarray(avg_distances)


def get_distances_excluding_self(X_train_t, X_val_t):
	n_tiss = X_val_t.shape[0]

	min_distances = []
	avg_distances = []
	for tiss_iter in range(n_tiss):
		distances = np.sum(np.square(np.delete(X_train_t, tiss_iter, axis=0) - X_val_t[tiss_iter,:]),axis=1)

		min_distances.append(np.min(distances))
		avg_distances.append(np.mean(distances))

	return np.asarray(min_distances), np.asarray(avg_distances)



################################################################################
# main
################################################################################
def main():
	######################
	# Command line args
	######################
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
	parser.add_argument('--n_validation_tissues', default=5, type=int,
						help='Number of validation tissues')

	# Defaults
	parser.add_argument('--KK', default=10, type=int,
						help='Number of dimensions')
	parser.add_argument('--tissue_mlp_hidden', default=64, type=int,
						help='Hidden units in tissue MLP')
	parser.add_argument('--tissue_mlp_dropout_rate', default=0.0, type=float,
						help='Dropout rate in tissue MLP')
	parser.add_argument('--learning_rate', default=1e-3, type=float,
						help='Learning rate')
	parser.add_argument('--num_epochs', default=200, type=int,
						help='Number of epochs')
	parser.add_argument('--batch_size', default=2048, type=int,
						help='Batch size')
	parser.add_argument('--random_seed', default=1, type=int,
						help='Base random seed')

	# NEW: restarts
	parser.add_argument('--n_restarts', default=3, type=int,
						help='Number of random restarts (different initializations)')

	args = parser.parse_args()

	# Validation: number of rows to use (can be < N to save time)
	N_VAL_ROWS_MAX = 20000000000

	# ---------------------------
	# Load data (once)
	# ---------------------------
	np.random.seed(args.random_seed + 2)

	# Load in tissue names
	tissue_names = load_in_ordered_tissue_names(args.tissue_file, args.expression_file)

	# Load in estimated eqtl effects
	eqtl_beta_hat, variant_names_beta, tissue_names_beta = load_in_eqtl_data(args.eqtl_effect_size_file)
	print('loaded effects')
	eqtl_beta_hat_se, variant_names_se, tissue_names_se = load_in_eqtl_data(args.eqtl_se_file)
	print('loaded ses')

	# Load in expression data
	expression_mat = (np.loadtxt(args.expression_file, dtype=str, delimiter='\t')[1:, :][:, 1:]).astype(float)

	# Split into training, validation, and test tissues
	orig_test_tissues = np.asarray(args.test_tissue_list.split(';'))
	train_tissue_idx, val_tissue_idx, test_tissue_idx = get_training_validation_and_test_tissue_indices(
		tissue_names, orig_test_tissues, args.n_validation_tissues
	)
	train_tissues = tissue_names[train_tissue_idx]
	val_tissues = tissue_names[val_tissue_idx]
	test_tissues = tissue_names[test_tissue_idx]

	# Slice eQTL data into TRAIN, VAL, and TEST sets
	tmp_beta_train = eqtl_beta_hat[:, train_tissue_idx]   # (N, T_train)
	valid_variant_gene_pairs = np.sum(np.isnan(tmp_beta_train) == False, axis=1) > 0

	beta_train = eqtl_beta_hat[:, train_tissue_idx][valid_variant_gene_pairs, :]
	se_train   = eqtl_beta_hat_se[:, train_tissue_idx][valid_variant_gene_pairs, :]
	X_train_t  = np.transpose(expression_mat[:, train_tissue_idx])

	beta_val   = eqtl_beta_hat[:, val_tissue_idx][valid_variant_gene_pairs, :]
	se_val     = eqtl_beta_hat_se[:, val_tissue_idx][valid_variant_gene_pairs, :]
	X_val_t    = np.transpose(expression_mat[:, val_tissue_idx])

	beta_test  = eqtl_beta_hat[:, test_tissue_idx][valid_variant_gene_pairs, :]
	se_test    = eqtl_beta_hat_se[:, test_tissue_idx][valid_variant_gene_pairs, :]
	X_test_t   = np.transpose(expression_mat[:, test_tissue_idx])

	# Masks of observed cells
	mask_train = ~np.isnan(beta_train)
	mask_val   = ~np.isnan(beta_val)
	mask_test  = ~np.isnan(beta_test)

	# Replace NaNs
	beta_train = np.nan_to_num(beta_train, nan=0.0).astype(np.float32)
	se_train   = np.nan_to_num(se_train,   nan=1.0).astype(np.float32)
	beta_val   = np.nan_to_num(beta_val,   nan=0.0).astype(np.float32)
	se_val     = np.nan_to_num(se_val,     nan=1.0).astype(np.float32)
	beta_test  = np.nan_to_num(beta_test,  nan=0.0).astype(np.float32)
	se_test    = np.nan_to_num(se_test,    nan=1.0).astype(np.float32)

	mask_train = mask_train.astype(np.bool_)
	mask_val   = mask_val.astype(np.bool_)
	mask_test  = mask_test.astype(np.bool_)
	X_train_t  = X_train_t.astype(np.float32)
	X_val_t    = X_val_t.astype(np.float32)
	X_test_t   = X_test_t.astype(np.float32)

	train_min_distances, train_avg_distances = get_distances_excluding_self(X_train_t, X_train_t)
	val_min_distances, val_avg_distances = get_distances(X_train_t, X_val_t)
	test_min_distances, test_avg_distances = get_distances(X_train_t, X_test_t)

	X_train_t = np.hstack((X_train_t, train_min_distances.reshape(-1,1), train_avg_distances.reshape(-1,1)))
	X_val_t = np.hstack((X_val_t, val_min_distances.reshape(-1,1), val_avg_distances.reshape(-1,1)))
	X_test_t = np.hstack((X_test_t, test_min_distances.reshape(-1,1), test_avg_distances.reshape(-1,1)))


	means = np.mean(X_train_t,axis=0)
	sdevs = np.std(X_train_t,axis=0)




	X_train_t = (X_train_t - means)/sdevs
	X_val_t = (X_val_t - means)/sdevs
	X_test_t= (X_test_t - means)/sdevs

	X_train_t = X_train_t[:, sdevs!=0.0]
	X_val_t = X_val_t[:, sdevs!=0.0]
	X_test_t = X_test_t[:, sdevs!=0.0]



	X_train_t[np.isnan(X_train_t)] = 0.0
	X_val_t[np.isnan(X_val_t)] = 0.0
	X_test_t[np.isnan(X_test_t)] = 0.0

	# ---- Move big arrays to CPU ----
	with tf.device("/CPU:0"):
		beta_train_tf = tf.constant(beta_train)
		se_train_tf   = tf.constant(se_train)
		beta_val_tf   = tf.constant(beta_val)
		se_val_tf     = tf.constant(se_val)
		mask_train_tf = tf.constant(mask_train)
		mask_val_tf   = tf.constant(mask_val)

	X_val_t_tf = tf.constant(X_val_t, dtype=tf.float32)

	NN = beta_train.shape[0]
	GG = X_val_t.shape[1]

	# ============================================================
	#  RESTART LOOP: train n_restarts times, keep best by val_corr
	# ============================================================

	best_overall_seed = None
	best_overall_epoch = None
	best_overall_val_corr = -np.inf
	best_overall_val_loss2 = np.inf
	best_overall_weights = None

	all_rows = tf.range(NN, dtype=tf.int32)
	AUTOTUNE = tf.data.AUTOTUNE

	for rr in range(args.n_restarts):
		seed = int(args.random_seed + rr)

		print("\n" + "=" * 80)
		print(f"RESTART {rr+1}/{args.n_restarts}  (seed={seed})")
		print("=" * 80)

		# Reset TF state between restarts
		tf.keras.backend.clear_session()

		# Seeds: make init + shuffles reproducible and different across restarts
		np.random.seed(seed + 2)
		tf.random.set_seed(seed + 3)

		# Build datasets over rows (rebuild each restart)
		train_ds = (
			tf.data.Dataset.from_tensor_slices(all_rows)
			.shuffle(buffer_size=min(NN, 10_000_000), seed=seed + 11, reshuffle_each_iteration=True)
			.batch(args.batch_size)
			.prefetch(AUTOTUNE)
		)

		N_val_rows = min(NN, N_VAL_ROWS_MAX)
		val_rows = tf.random.shuffle(all_rows, seed=seed + 17)[:N_val_rows]

		val_ds = (
			tf.data.Dataset.from_tensor_slices(val_rows)
			.batch(args.batch_size)
			.prefetch(AUTOTUNE)
		)

		# Instantiate fresh model & trainer
		model = EQTLFactorModel(
			N=NN,
			G=GG,
			K=args.KK,
			X_train_t=X_train_t,
			hidden=args.tissue_mlp_hidden,
			dropout_rate=args.tissue_mlp_dropout_rate,
		)

		trainer = EQTLTrainer(
			model=model,
			beta_train_tf=beta_train_tf,
			se_train_tf=se_train_tf,
			mask_train_tf=mask_train_tf,
			beta_val_tf=beta_val_tf,
			se_val_tf=se_val_tf,
			mask_val_tf=mask_val_tf,
			X_val_t_tf=X_val_t_tf,
			batch_size=args.batch_size,
			learning_rate=args.learning_rate,
		)

		# Train and get best-epoch stats for this restart
		trained_model, best_epoch, best_val_loss2, best_val_corr = trainer.fit(
			train_ds, val_ds, num_epochs=args.num_epochs
		)

		print(f"Restart {rr+1} summary: best_epoch={best_epoch}, best_val_corr={best_val_corr:.6f}, best_val_loss2={best_val_loss2:.6f}")

		# BEST RESTART = LOWEST VAL LOSS
		if best_val_loss2 < best_overall_val_loss2:
			best_overall_val_loss2 = best_val_loss2
			best_overall_val_corr = best_val_corr
			best_overall_epoch = best_epoch
			best_overall_seed = seed
			best_overall_weights = trained_model.get_weights()
			print(">>> NEW BEST OVERALL <<<")

	# Rebuild best model and load best weights (clean final model)
	tf.keras.backend.clear_session()
	best_model = EQTLFactorModel(
		N=NN,
		G=GG,
		K=args.KK,
		X_train_t=X_train_t,
		hidden=args.tissue_mlp_hidden,
		dropout_rate=args.tissue_mlp_dropout_rate,
	)
	# IMPORTANT: build variables (subclassed model) before set_weights
	_ = best_model(tf.constant([0], dtype=tf.int32), training=False)

	best_model.set_weights(best_overall_weights)
	print("\n" + "#" * 80)
	print(f"BEST OVERALL: seed={best_overall_seed}, epoch={best_overall_epoch}, val_corr={best_overall_val_corr:.6f}, val_loss2={best_overall_val_loss2:.6f}")
	print("#" * 80 + "\n")

	# Save best-restart summary
	best_summary_file = args.output_stem + "_best_restart_summary.txt"
	with open(best_summary_file, "w") as f:
		f.write(f"best_seed\t{best_overall_seed}\n")
		f.write(f"best_epoch\t{best_overall_epoch}\n")
		f.write(f"best_val_corr\t{best_overall_val_corr}\n")
		f.write(f"best_val_loss2\t{best_overall_val_loss2}\n")
	print(best_summary_file)

	# ---------------------------
	# Test evaluation (best model)
	# ---------------------------
	test_losses, test_losses_ses, test_corrz, test_corrz_ses, test_beta_preds, test_betas, test_beta_se, test_beta_pred_var = get_test_losses(
		best_model, beta_test, se_test, mask_test, X_test_t
	)


	# Write test losses
	test_loss_output_file = args.output_stem + '_test_loss_summary.txt'
	t = open(test_loss_output_file, 'w')
	t.write('tissue\tloss\tloss_se\tcorrelation\tcorrelation_se\n')
	for ii, tissue_name in enumerate(test_tissues):
		t.write(
			tissue_name + '\t' +
			str(test_losses[ii]) + '\t' +
			str(test_losses_ses[ii]) + '\t' +
			str(test_corrz[ii]) + '\t' +
			str(test_corrz_ses[ii]) + '\n'
		)
	t.close()

	# Write per-cell predictions (as in your original code)
	test_pred_output_file = args.output_stem + '_test_preds.txt'
	t = open(test_pred_output_file, 'w')
	t.write('beta\tbeta_se\tpred_beta\tpred_beta_var\n')
	for ii, test_beta_pred in enumerate(test_beta_preds):
		t.write(
			str(test_betas[ii]) + '\t' +
			str(test_beta_se[ii]) + '\t' +
			str(test_beta_pred) + '\t' +
			str(test_beta_pred_var) + '\n'
		)
	t.close()
	print(test_pred_output_file)

	return


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
	main()
