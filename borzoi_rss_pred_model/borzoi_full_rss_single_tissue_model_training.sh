#!/bin/bash
#SBATCH -t 0-15:30                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                        # Partition to run in
#SBATCH --mem=8GB 


source ~/.bashrc
conda activate borzoi

gtex_tissue_names_file="$1"
prediction_input_data_summary_filestem="$2"
test_tissue="$3"
model_training_output_stem="$4"
learning_rate="$5"
l2_variant_reg_strength="$6"
variant_encoder_architecture="$7"

echo $model_training_output_stem

date
python "borzoi_full_rss_single_tissue_model_training.py" \
	--gtex-tissue-names-file "$gtex_tissue_names_file" \
	--prediction-input-data-summary-filestem "$prediction_input_data_summary_filestem" \
	--test-tissue "$test_tissue" \
	--model-training-output-stem "$model_training_output_stem" \
	--learning-rate "$learning_rate" \
	--l2-variant-reg-strength "$l2_variant_reg_strength" \
	--variant-encoder-architecture "$variant_encoder_architecture"
date
