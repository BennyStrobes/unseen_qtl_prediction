#########################
# Input data
#########################
# GTEx v10 expression TPM file
gtex_tpm_expression="/lab-share/CHIP-Strober-e2/Public/GTEx/expression/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz"

# GTEx v10 per tissue expression dir
gtex_per_tissue_expression_dir="/lab-share/CHIP-Strober-e2/Public/GTEx/expression/per_tissue_expression/"

# GTEx sample attributes files
# Contains tissue identity information
gtex_sample_attributes_file="/lab-share/CHIP-Strober-e2/Public/GTEx/gtex_sample_attributes/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt"

# 1000G genotype dir
genotype_dir_1000_G="/lab-share/CHIP-Strober-e2/Public/1000G_Phase3/hg38/"

# Gencode gene annotation file
gencode_gene_annotation_file="/lab-share/CHIP-Strober-e2/Public/gene_annotation_files/gencode.v39.gtex.protein_coding.genes.gtf"

# Data processed for input to prediction
prediction_input_data_summary_filestem="/lab-share/CHIP-Strober-e2/Public/ben/s2e_uncertainty/eqtl_borzoi_integration_processed_data/borzoi/gene_borzoi_summary_chr"
prediction_inv_ld_input_data_summary_filestem="/lab-share/CHIP-Strober-e2/Public/ben/s2e_uncertainty/eqtl_borzoi_integration_processed_data/borzoi/gene_borzoi_summary_big_inv_ld_ready_chr"

# Ordered gtex tissues names 
gtex_tissue_names_file="/lab-share/CHIP-Strober-e2/Public/ben/s2e_uncertainty/eqtl_borzoi_integration_processed_data/borzoi/ordered_gtex_tissues_chr1.txt"

# Borzoi target fiel
borzoi_target_file="/lab-share/CHIP-Strober-e2/Public/ben/s2e_uncertainty/borzoi_input_data/models/targets_human.txt"

#########################
# Output data
#########################
# Output root directory
output_root_dir="/lab-share/CHIP-Strober-e2/Public/ben/unseen_qtl_prediction/causal_eqtl_lf_model/"

# Output directory processed summary statistics
tissue_names_dir=${output_root_dir}"tissue_names/"

# Output directory containing processed expression
processed_expression_dir=${output_root_dir}"processed_gene_expression/"

# Training data input directory
model_training_dir=${output_root_dir}"model_training/"

# Training data input directory
visualization_dir=${output_root_dir}"visualization/"



#########################
# Scripts
#########################

##########
# Process gene expression levels
processed_all_sample_expression_file=${processed_expression_dir}"log_transformed_tpm_expression_all_samples.txt"
processed_all_sample_sample_names_file=${processed_expression_dir}"log_transformed_tpm_expression_all_sample_names.txt"
if false; then
sh process_tpm_expression_levels.sh $gtex_tpm_expression $gtex_tissue_names_file $gtex_per_tissue_expression_dir $gencode_gene_annotation_file $gtex_sample_attributes_file $processed_all_sample_expression_file $processed_all_sample_sample_names_file
fi


##########
# Randomly select single training sample and single test sample for model fitting
seed="1"
single_samp_per_tissue_expr_file=${processed_expression_dir}"single_sample_per_tissue_expression_seed"${seed}".txt"
if false; then
source ~/.bashrc
conda activate borzoi
python randomly_select_single_sample_for_each_tissue.py $gtex_tissue_names_file $seed $processed_all_sample_expression_file $single_samp_per_tissue_expr_file
fi

##########
# Process target file
target_summary_file=${tissue_names_dir}"borzoi_target_summary.txt"
borzoi_gtex_tissues_file=${tissue_names_dir}"borzoi_gtex_tissues.txt"
if false; then
source ~/.bashrc
conda activate borzoi
python process_borzoi_target_files.py $borzoi_target_file $gtex_tissue_names_file $gtex_sample_attributes_file $target_summary_file $borzoi_gtex_tissues_file
fi



########################################
# Hold out single tissue model training
########################################

# Sweep over test tissue (from borzoi_gtex_tissues_file), learning rate, number of latent factors, tissue-encoder L2, and variant (U) L1
if false; then
for test_tissue in $(tail -n +2 $borzoi_gtex_tissues_file | cut -f1); do
for learning_rate in "1e-3" "5e-3" "1e-2"; do
for n_factors in "10"; do
for l2_tissue_reg_strength in "1e-4"; do
for l1_variant_reg_strength in "1e-4"; do
	model_training_output_stem=${model_training_dir}"full_rss_lf_model_train_test_tissue_"${test_tissue}"_lr_"${learning_rate}"_l2t_"${l2_tissue_reg_strength}"_l1v_"${l1_variant_reg_strength}"_var_arch_K_"${n_factors}
	sbatch borzoi_full_rss_lf_model_training.sh $gtex_tissue_names_file $single_samp_per_tissue_expr_file $prediction_inv_ld_input_data_summary_filestem $test_tissue $model_training_output_stem $learning_rate $l2_tissue_reg_strength $l1_variant_reg_strength $n_factors
done
done
done
done
done
fi






########################################
# Summarize training logs + evaluate best model per tissue (one batch job per tissue)
########################################
if false; then
for test_tissue in $(tail -n +2 $borzoi_gtex_tissues_file | cut -f1); do
	sbatch summarize_and_evaluate_lf_model.sh $gtex_tissue_names_file $single_samp_per_tissue_expr_file $prediction_inv_ld_input_data_summary_filestem $test_tissue $model_training_dir
done
fi

