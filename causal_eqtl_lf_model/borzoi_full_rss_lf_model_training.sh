#!/bin/bash
#SBATCH -t 0-15:30                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                        # Partition to run in
#SBATCH --mem=8GB 


source ~/.bashrc
conda activate borzoi

gtex_tissue_names_file="$1"
single_samp_per_tissue_expr_file="$2"
prediction_input_data_summary_filestem="$3"
test_tissue="$4"
model_training_output_stem="$5"
learning_rate="$6"
l2_tissue_reg_strength="$7"
l1_variant_reg_strength="$8"
n_factors="$9"
# Ridge on the K x K Gram matrix in the closed-form U solve (fraction of its mean diagonal).
# Optional 10th arg; falls back to the script default when not given.
u_ridge_frac="${10:-1e-2}"

echo $gtex_tissue_names_file
echo $model_training_output_stem

date
if false; then
python "borzoi_full_rss_lf_model_training.py" \
	--gtex-tissue-names-file "$gtex_tissue_names_file" \
	--single-samp-per-tissue-expr-file "$single_samp_per_tissue_expr_file" \
	--prediction-input-data-summary-filestem "$prediction_input_data_summary_filestem" \
	--test-tissue "$test_tissue" \
	--model-training-output-stem "$model_training_output_stem" \
	--learning-rate "$learning_rate" \
	--l2-tissue-reg-strength "$l2_tissue_reg_strength" \
	--l1-variant-reg-strength "$l1_variant_reg_strength" \
	--n-factors "$n_factors"
date

fi
# Closed-form-U variant (U solved exactly given V each gene visit; only the encoder takes Adam steps).
# Writes to a separate output stem so the two runs do not overwrite each other.
python "borzoi_full_rss_lf_model_training_closed_form_U.py" \
	--gtex-tissue-names-file "$gtex_tissue_names_file" \
	--single-samp-per-tissue-expr-file "$single_samp_per_tissue_expr_file" \
	--prediction-input-data-summary-filestem "$prediction_input_data_summary_filestem" \
	--test-tissue "$test_tissue" \
	--model-training-output-stem "${model_training_output_stem}_closed_form_U" \
	--learning-rate "$learning_rate" \
	--l2-tissue-reg-strength "$l2_tissue_reg_strength" \
	--l1-variant-reg-strength "$l1_variant_reg_strength" \
	--u-ridge-frac "$u_ridge_frac" \
	--n-factors "$n_factors"
date
