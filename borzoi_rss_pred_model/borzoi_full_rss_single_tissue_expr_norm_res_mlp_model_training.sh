#!/bin/bash
#SBATCH -t 0-30:30                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                         # Partition to run in
#SBATCH --mem=15GB  



source ~/.bashrc
conda activate borzoi

gtex_tissue_names_file="$1"
prediction_input_data_summary_filestem="$2"
test_tissue="$3"
model_training_output_stem="$4"
learning_rate="$5"
l2_variant_reg_strength="$6"
variant_encoder_architecture="$7"
gtex_tpm_expression="${8}"
gtex_sample_attributes_file="${9}"
res_mlp_blocks_per_stage="${10}"
res_mlp_dropout_rate="${11}"

echo $model_training_output_stem

date
python "borzoi_full_rss_single_tissue_expr_norm_model_training.py" \
	--gtex-tissue-names-file "$gtex_tissue_names_file" \
	--prediction-input-data-summary-filestem "$prediction_input_data_summary_filestem" \
	--expression-tpm-file "$gtex_tpm_expression" \
	--expression-sample-file "$gtex_sample_attributes_file" \
	--test-tissue "$test_tissue" \
	--model-training-output-stem "$model_training_output_stem" \
	--learning-rate "$learning_rate" \
	--l2-variant-reg-strength "$l2_variant_reg_strength" \
	--variant-encoder-type "res_mlp" \
	--variant-encoder-architecture "$variant_encoder_architecture" \
	--res-mlp-blocks-per-stage "$res_mlp_blocks_per_stage" \
	--res-mlp-dropout-rate "$res_mlp_dropout_rate"
date
