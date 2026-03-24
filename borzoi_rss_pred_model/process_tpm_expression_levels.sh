#!/bin/bash
#SBATCH -t 0-0:30                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                        # Partition to run in
#SBATCH --mem=5GB 



gtex_tpm_expression="${1}"
gtex_tissue_names_file="${2}"
gtex_per_tissue_expression_dir="${3}"
gencode_gene_annotation_file="${4}"
gtex_sample_attributes_file="${5}"
processed_all_sample_expression_file="${6}"
processed_all_sample_sample_names_file="${7}"


source ~/.bashrc
conda activate borzoi

python process_tpm_expression_levels.py $gtex_tpm_expression $gtex_tissue_names_file $gtex_per_tissue_expression_dir $gencode_gene_annotation_file $gtex_sample_attributes_file $processed_all_sample_expression_file $processed_all_sample_sample_names_file