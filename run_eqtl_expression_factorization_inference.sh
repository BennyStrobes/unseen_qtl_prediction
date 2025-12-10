#!/bin/bash
#SBATCH -t 0-2:30                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                        # Partition to run in
#SBATCH --mem=10GB 



train_eqtl_effect_size_file="${1}"
train_eqtl_se_file="${2}"
single_samp_per_tissue_training_pc_file="${3}"
test_tissue="${4}"
gtex_tissue_names_file="${5}"
output_stem="${6}"
single_samp_per_tissue_pc_file="${7}"


source ~/.bashrc
conda activate borzoi

echo "Factorization"
python run_eqtl_expression_factorization_inference.py \
  --eqtl_effect_size_file "$train_eqtl_effect_size_file" \
  --eqtl_se_file "$train_eqtl_se_file" \
  --expression_file "$single_samp_per_tissue_training_pc_file" \
  --test_tissue_list $test_tissue \
  --tissue_file $gtex_tissue_names_file \
  --output_stem "$output_stem"


echo "Nearest neighbor"

python predict_eqtls_from_nearest_tissue.py \
  --eqtl_effect_size_file "$train_eqtl_effect_size_file" \
  --eqtl_se_file "$train_eqtl_se_file" \
  --expression_file "$single_samp_per_tissue_pc_file" \
  --test_tissue_list $test_tissue \
  --tissue_file $gtex_tissue_names_file \
  --output_stem "$output_stem"