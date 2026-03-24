#!/bin/bash
#SBATCH -t 0-10:30                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                        # Partition to run in
#SBATCH --mem=8GB 


source ~/.bashrc
conda activate borzoi

gtex_tissue_names_file="$1"
single_samp_per_tissue_expr_file="$2"
prediction_input_data_summary_file="$3"
test_tissue="$4"
model_training_output_stem="$5"

date
python "borzoi_rss_model_training.py" \
	--gtex-tissue-names-file "$gtex_tissue_names_file" \
	--single-samp-per-tissue-expr-file "$single_samp_per_tissue_expr_file" \
	--prediction-input-data-summary-file "$prediction_input_data_summary_file" \
	--test-tissue "$test_tissue" \
	--model-training-output-stem "$model_training_output_stem"
date