import argparse
import json
import numpy as np
import tensorflow as tf


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


def load_all_gene_data_map(prediction_input_data_summary_filestem):
	# gene_name -> (snp_summary_file, zed_file, N_eff_file, ld_file, inv_ld_file, borzoi_file, n_snps)
	# Unlike training's loader this does NO random subsampling and applies no min-SNP filter: we
	# evaluate exactly the genes that received a learned U block during training, matched by name,
	# so we just need every gene's file paths available for lookup.
	gene_map = {}
	for chrom_num in range(1, 23):
		f = open(prediction_input_data_summary_filestem + str(chrom_num) + '.txt')
		head_count = 0
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
			gene_map[gene_name] = (snp_summary_file, zed_file, N_eff_file, ld_file, inv_ld_file, borzoi_file, n_snps_per_gene)
		f.close()
	return gene_map


def find_best_model_stem(training_summary_file, model_training_dir):
	# training_summary columns (from summarize_lf_model_training_logs.py):
	# test_tissue  learning_rate  l2_tissue_reg_strength  l1_variant_reg_strength  n_factors
	#   best_fit_iter  train_loss  val_loss  val_corr
	# Return the model-output stem of the row with the lowest validation loss.
	head_count = 0
	best_val_loss = np.inf
	best_params = None
	f = open(training_summary_file)
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		if len(data) != 9:
			continue
		val_loss = float(data[7])
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_params = {'test_tissue': data[0], 'lr': data[1], 'l2t': data[2],
				'l1v': data[3], 'n_factors': data[4], 'val_loss': val_loss}
	f.close()
	if best_params is None:
		raise ValueError('No usable models found in training summary file: ' + training_summary_file)
	stem = model_training_dir + 'full_rss_lf_model_train_test_tissue_' + best_params['test_tissue'] + \
		'_lr_' + best_params['lr'] + '_l2t_' + best_params['l2t'] + \
		'_l1v_' + best_params['l1v'] + '_var_arch_K_' + best_params['n_factors']
	return stem, best_params


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gtex-tissue-names-file', required=True, type=str)
	parser.add_argument('--single-samp-per-tissue-expr-file', required=True, type=str)
	parser.add_argument('--prediction-input-data-summary-filestem', required=True, type=str)
	parser.add_argument('--test-tissue', required=True, type=str)
	parser.add_argument('--training-summary-file', required=True, type=str,
		help='Per-tissue summary from summarize_lf_model_training_logs.py; used to pick the lowest-val-loss config.')
	parser.add_argument('--model-training-dir', required=True, type=str,
		help='Directory holding the saved model artifacts (used to reconstruct the best config stem).')
	parser.add_argument('--evaluation-output-stem', required=True, type=str)
	return parser.parse_args()


def main():
	args = parse_args()

	# ------------------------------------------------------------------
	# Select and load the best (lowest validation loss) model for this tissue
	# ------------------------------------------------------------------
	best_model_stem, best_params = find_best_model_stem(args.training_summary_file, args.model_training_dir)
	print('Best model for test tissue ' + args.test_tissue + ': ' + best_model_stem +
		' (val_loss=' + str(best_params['val_loss']) + ')', flush=True)

	tissue_encoder = tf.keras.models.load_model(best_model_stem + '_tissue_encoder.keras')
	variant_embedding_table = np.load(best_model_stem + '_variant_embedding.npy')     # (n_total_variants, K)
	with open(best_model_stem + '_gene_to_u_block.json') as f:
		gene_to_u_block = json.load(f)                                                # gene_name -> [offset, n_snps]

	meta = np.load(best_model_stem + '_eval_metadata.npz')
	gene_expression_means = meta['gene_expression_means']                            # (n_expr_genes,)
	gene_expression_stds = meta['gene_expression_stds']                              # (n_expr_genes,)
	test_tissue_index = int(meta['test_tissue_index'])
	beta_scale = float(meta['beta_scale'])

	# ------------------------------------------------------------------
	# Encode the held-out test tissue -> its latent factor V_test
	# ------------------------------------------------------------------
	all_tissue_names = load_in_tissue_names(args.gtex_tissue_names_file)
	gene_expression_data, ge_tissue_names = load_in_expression_data(args.single_samp_per_tissue_expr_file)
	if np.array_equal(ge_tissue_names, all_tissue_names) == False:
		raise ValueError('Expression tissue-name order does not match the gtex tissue-names file')
	if all_tissue_names[test_tissue_index] != args.test_tissue:
		raise ValueError('Saved test_tissue_index points at ' + all_tissue_names[test_tissue_index] +
			' but --test-tissue is ' + args.test_tissue)

	# Standardize the test tissue's expression with the TRAIN-tissue per-gene mean/std, exactly as
	# training standardized its inputs, then run the encoder to get V_test.
	test_tissue_expression = gene_expression_data[:, test_tissue_index]
	test_tissue_expression_std = (test_tissue_expression - gene_expression_means) / gene_expression_stds
	test_tissue_expression_tf = tf.convert_to_tensor(test_tissue_expression_std[None, :].astype(np.float32))
	V_test = tissue_encoder(test_tissue_expression_tf, training=False).numpy().astype(np.float64)     # (1, K)

	# ------------------------------------------------------------------
	# Per-gene / per-variant evaluation on the test tissue
	# ------------------------------------------------------------------
	gene_data_map = load_all_gene_data_map(args.prediction_input_data_summary_filestem)

	gene_output_file = args.evaluation_output_stem + '_test_tissue_gene_evaluation.txt'
	variant_output_file = args.evaluation_output_stem + '_test_tissue_variant_gene_pair_evaluation.txt'

	n_genes_written = 0
	n_genes_skipped = 0
	with open(gene_output_file, 'w') as gf, open(variant_output_file, 'w') as vf:
		gf.write('gene_name\tn_snps\tloss\tmarginal_std_effect_corr\texpr_corr\n')
		vf.write('gene_name\tvariant_name\tobs_marginal_std_effect\tpred_marginal_std_effect\tstandard_error\n')

		for gene_name in gene_to_u_block:
			if gene_name not in gene_data_map:
				print('Warning: gene ' + gene_name + ' has a learned U block but no data files; skipping', flush=True)
				n_genes_skipped = n_genes_skipped + 1
				continue

			snp_summary_file, zed_file, N_eff_file, ld_file, inv_ld_file, borzoi_file, n_gene_snps = gene_data_map[gene_name]
			u_offset, u_n_snps = gene_to_u_block[gene_name]

			gene_LD = np.load(ld_file).astype(np.float64)
			gene_inv_LD = np.load(inv_ld_file).astype(np.float64)
			gene_zeds = np.load(zed_file)[:, test_tissue_index].astype(np.float64)          # (n_snps,) observed z-scores
			gene_N_eff = np.load(N_eff_file)[:, test_tissue_index].astype(np.float64)       # (n_snps,) effective sample size
			gene_snp_summary = np.loadtxt(snp_summary_file, dtype=str)
			gene_variant_names = gene_snp_summary[1:, 0]

			# Every SNP-dim quantity (LD, U block, z-scores, variant names) must agree in length.
			if not (gene_LD.shape[0] == u_n_snps == gene_LD.shape[1] == len(gene_zeds) == len(gene_variant_names)):
				print('Warning: SNP-dimension mismatch for gene ' + gene_name + '; skipping', flush=True)
				n_genes_skipped = n_genes_skipped + 1
				continue

			U_g = variant_embedding_table[u_offset:u_offset + u_n_snps, :].astype(np.float64)   # (n_snps, K)

			# Predicted standardized causal effect per SNP for the test tissue: beta = c * U_g . V_test^T
			beta_causal_std = beta_scale * (U_g @ V_test.T)[:, 0]                              # (n_snps,)
			# LD-propagated (marginal) standardized effect, and the implied marginal z-score.
			R_beta = gene_LD @ beta_causal_std                                                # (n_snps,)
			pred_marginal_std = R_beta
			pred_z = np.sqrt(gene_N_eff) * pred_marginal_std

			obs_z = gene_zeds
			# Standard error of the marginal standardized effect (~1/sqrt(N)); obs effect = z / sqrt(N).
			standard_error = 1.0 / np.sqrt(gene_N_eff)
			obs_marginal_std = obs_z * standard_error

			# ---- per variant-gene pair (always length n_snps; observed cols are NaN if the tissue is missing) ----
			for variant_name, obs_eff, pred_eff, se in zip(gene_variant_names, obs_marginal_std, pred_marginal_std, standard_error):
				vf.write(gene_name + '\t' + variant_name + '\t' + str(obs_eff) + '\t' + str(pred_eff) + '\t' + str(se) + '\n')

			# ---- per gene ----
			# Missingness is all-or-nothing per (gene, tissue); if this tissue is unobserved, report NaNs.
			if np.isnan(obs_z).any() or np.isnan(gene_N_eff).any():
				gf.write(gene_name + '\t' + str(n_gene_snps) + '\tnan\tnan\tnan\n')
				n_genes_written = n_genes_written + 1
				continue

			# RSS quadratic-form loss (matches training's forward_gene): resid^T inv_LD resid on z-scores.
			residuals = obs_z - pred_z
			gene_loss = float(residuals @ (gene_inv_LD @ residuals))

			# Correlation of observed vs predicted marginal standardized effects.
			if len(obs_marginal_std) > 1 and np.std(obs_marginal_std) > 0 and np.std(pred_marginal_std) > 0:
				marginal_std_effect_corr = float(np.corrcoef(obs_marginal_std, pred_marginal_std)[0, 1])
			else:
				marginal_std_effect_corr = np.nan

			# Correlation of predicted vs observed (standardized) expression, from summary statistics:
			#   pred_expr = X beta_causal, obs_expr ~ unit variance
			#   corr = (beta_causal . obs_marginal_std) / sqrt(beta_causal^T R beta_causal)
			pred_expr_variance = float(beta_causal_std @ R_beta)
			if pred_expr_variance > 0:
				expr_corr = float((obs_marginal_std @ beta_causal_std) / np.sqrt(pred_expr_variance))
			else:
				expr_corr = np.nan

			gf.write(gene_name + '\t' + str(n_gene_snps) + '\t' + str(gene_loss) + '\t' +
				str(marginal_std_effect_corr) + '\t' + str(expr_corr) + '\n')
			n_genes_written = n_genes_written + 1

	print('Wrote ' + str(n_genes_written) + ' genes (skipped ' + str(n_genes_skipped) + ') for test tissue ' + args.test_tissue, flush=True)
	print('  gene-level evaluation:        ' + gene_output_file, flush=True)
	print('  variant-gene-pair evaluation: ' + variant_output_file, flush=True)


if __name__ == '__main__':
	main()
