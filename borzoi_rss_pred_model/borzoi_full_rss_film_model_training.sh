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
l2_variant_reg_strength="$8"
variant_encoder_architecture="$9"

echo $model_training_output_stem

date
python "borzoi_full_rss_film_model_training.py" \
	--gtex-tissue-names-file "$gtex_tissue_names_file" \
	--single-samp-per-tissue-expr-file "$single_samp_per_tissue_expr_file" \
	--prediction-input-data-summary-filestem "$prediction_input_data_summary_filestem" \
	--test-tissue "$test_tissue" \
	--model-training-output-stem "$model_training_output_stem" \
	--learning-rate "$learning_rate" \
	--l2-tissue-reg-strength "$l2_tissue_reg_strength" \
	--l2-variant-reg-strength "$l2_variant_reg_strength" \
	--variant-encoder-architecture "$variant_encoder_architecture"
date
