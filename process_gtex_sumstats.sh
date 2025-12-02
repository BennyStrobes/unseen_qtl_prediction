#!/bin/bash
#SBATCH -t 0-0:30                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                        # Partition to run in
#SBATCH --mem=5GB 





tissue_name="${1}"
processed_gtex_sumstats_dir="${2}"
gtex_summary_stats_dir="${3}"
genotype_dir_1000_G="${4}"
cis_window="${5}"
snp_set="${6}"
gtex_per_tissue_expression_dir="${7}"

source ~/.bashrc
conda activate borzoi

python reprocess_gtex_sumstats.py $tissue_name $processed_gtex_sumstats_dir $gtex_summary_stats_dir $genotype_dir_1000_G $cis_window $snp_set $gtex_per_tissue_expression_dir