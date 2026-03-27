#!/bin/bash
#SBATCH -t 0-0:30                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                        # Partition to run in
#SBATCH --mem=8GB 



gtex_tissue_name="${1}"
borzoi_target_index="${2}"
borzoi_eval_output_stem="${3}"
prediction_inv_ld_input_data_summary_filestem="${4}"
gtex_tissue_names_file="${5}"

source ~/.bashrc
conda activate borzoi

echo $gtex_tissue_name

python borzoi_pred_evaluation.py $gtex_tissue_name $borzoi_target_index $borzoi_eval_output_stem $prediction_inv_ld_input_data_summary_filestem $gtex_tissue_names_file
