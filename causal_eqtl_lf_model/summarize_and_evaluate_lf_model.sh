#!/bin/bash
#SBATCH -t 0-01:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                        # Partition to run in
#SBATCH --mem=8GB


source ~/.bashrc
conda activate borzoi

gtex_tissue_names_file="$1"
single_samp_per_tissue_expr_file="$2"
prediction_input_data_summary_filestem="$3"
test_tissue="$4"
model_training_dir="$5"

# Derived per-tissue paths (match the naming used elsewhere in the pipeline)
model_training_summary_file=${model_training_dir}"training_summary_test_tissue_"${test_tissue}".txt"
evaluation_output_stem=${model_training_dir}"best_model_test_tissue_evaluation_"${test_tissue}

echo $test_tissue
echo $model_training_summary_file
echo $evaluation_output_stem

date
# 1) Summarize this tissue's training logs -> rank configs by validation loss
python "summarize_lf_model_training_logs.py" \
	"$test_tissue" \
	"$model_training_dir" \
	"$model_training_summary_file"

# 2) Evaluate the best (lowest val loss) model on held-out test-tissue data
python "evaluate_lf_model_on_test_tissue.py" \
	--gtex-tissue-names-file "$gtex_tissue_names_file" \
	--single-samp-per-tissue-expr-file "$single_samp_per_tissue_expr_file" \
	--prediction-input-data-summary-filestem "$prediction_input_data_summary_filestem" \
	--test-tissue "$test_tissue" \
	--training-summary-file "$model_training_summary_file" \
	--model-training-dir "$model_training_dir" \
	--evaluation-output-stem "$evaluation_output_stem"
date
