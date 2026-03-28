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
prediction_inv_ld_input_data_summary_filestem="/lab-share/CHIP-Strober-e2/Public/ben/s2e_uncertainty/eqtl_borzoi_integration_processed_data/borzoi/gene_borzoi_summary_inv_ld_ready_chr"

# Ordered gtex tissues names 
gtex_tissue_names_file="/lab-share/CHIP-Strober-e2/Public/ben/s2e_uncertainty/eqtl_borzoi_integration_processed_data/borzoi/ordered_gtex_tissues_chr1.txt"

# Borzoi target fiel
borzoi_target_file="/lab-share/CHIP-Strober-e2/Public/ben/s2e_uncertainty/borzoi_input_data/models/targets_human.txt"

#########################
# Output data
#########################
# Output root directory
output_root_dir="/lab-share/CHIP-Strober-e2/Public/ben/unseen_qtl_prediction/borzoi_rss_pred_model/"

# Output directory processed summary statistics
tissue_names_dir=${output_root_dir}"tissue_names/"

# Output directory containing processed expression
processed_expression_dir=${output_root_dir}"processed_gene_expression/"

# Training data input directory
model_training_dir=${output_root_dir}"modeling_training/"

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
if false; then
learning_rate="1e-5"
l2_tissue_reg_strength="100.0"
l2_variant_reg_strength="100.0"
test_tissue="Adipose_Visceral_Omentum"
	model_training_output_stem=${model_training_dir}"full_rss_model_held_out_genes_eval_train_test_tissue_"${test_tissue}"_lr_"${learning_rate}"_l2t_"${l2_tissue_reg_strength}"_l2v_"${l2_variant_reg_strength}"_var_arch_"${variant_encoder_architecture//,/x}
	sh borzoi_full_rss_model_training.sh $gtex_tissue_names_file $single_samp_per_tissue_expr_file $prediction_inv_ld_input_data_summary_filestem $test_tissue $model_training_output_stem $learning_rate $l2_tissue_reg_strength $l2_variant_reg_strength $variant_encoder_architecture
fi

if false; then
tail -n +2 $gtex_tissue_names_file | while read -r test_tissue; do
	learning_rate="1e-5"
	l2_tissue_reg_strength="100.0"
	l2_variant_reg_strength="100.0"
	model_training_output_stem=${model_training_dir}"full_rss_model_held_out_genes_eval_train_test_tissue_"${test_tissue}"_lr_"${learning_rate}"_l2t_"${l2_tissue_reg_strength}"_l2v_"${l2_variant_reg_strength}"_var_arch_"${variant_encoder_architecture//,/x}
	sbatch borzoi_full_rss_model_training.sh $gtex_tissue_names_file $single_samp_per_tissue_expr_file $prediction_inv_ld_input_data_summary_filestem $test_tissue $model_training_output_stem $learning_rate $l2_tissue_reg_strength $l2_variant_reg_strength $variant_encoder_architecture
done
fi

if false; then
tail -n +2 $gtex_tissue_names_file | while read -r test_tissue; do
	learning_rate="1e-5"
	l2_tissue_reg_strength="100.0"
	l2_variant_reg_strength="100.0"
	variant_encoder_architecture="2048,1024,512,256,128,64,32"
	model_training_output_stem=${model_training_dir}"full_rss_model_held_out_genes_eval_train_test_tissue_"${test_tissue}"_lr_"${learning_rate}"_l2t_"${l2_tissue_reg_strength}"_l2v_"${l2_variant_reg_strength}"_var_arch_"${variant_encoder_architecture//,/x}
	sbatch borzoi_full_rss_model_training.sh $gtex_tissue_names_file $single_samp_per_tissue_expr_file $prediction_inv_ld_input_data_summary_filestem $test_tissue $model_training_output_stem $learning_rate $l2_tissue_reg_strength $l2_variant_reg_strength $variant_encoder_architecture
done
fi


##########
# Evaluate standard borzoi predictions
if false; then
tail -n +2 "$borzoi_gtex_tissues_file" | while IFS=$'\t' read -r test_tissue borzoi_target_index; do
	borzoi_eval_output_stem=${model_training_dir}"borzoi_eval_"${test_tissue}
    sbatch borzoi_pred_evaluation.sh $test_tissue $borzoi_target_index $borzoi_eval_output_stem $prediction_inv_ld_input_data_summary_filestem $gtex_tissue_names_file
done
fi


########################################
# Single tissue analysis
########################################
test_tissue="Adipose_Subcutaneous"
learning_rates=("1e-5" "1e-4" "1e-3")
l2_variant_reg_strengths=("0.0" "1.0" "100.0")
variant_encoder_architectures=("2048,1024,512,256,128,64,32" "256,128,64,32")
if false; then
for learning_rate in "${learning_rates[@]}"; do
for l2_variant_reg_strength in "${l2_variant_reg_strengths[@]}"; do
for variant_encoder_architecture in "${variant_encoder_architectures[@]}"; do
	model_training_output_stem=${model_training_dir}"full_rss_model_single_tissue_"${test_tissue}"_lr_"${learning_rate}"_l2t_NA_l2v_"${l2_variant_reg_strength}"_var_arch_"${variant_encoder_architecture//,/x}
	sbatch borzoi_full_rss_single_tissue_model_training.sh $gtex_tissue_names_file $prediction_inv_ld_input_data_summary_filestem $test_tissue $model_training_output_stem $learning_rate $l2_variant_reg_strength $variant_encoder_architecture
done
done
done
fi



test_tissue="Adipose_Subcutaneous"
learning_rates=("1e-5" "1e-4" "1e-3")
l2_variant_reg_strengths=("0.0" "1.0" "100.0")
variant_encoder_architectures=("2048,1024,512,256,128,64,32" "256,128,64,32")
if false; then
for learning_rate in "${learning_rates[@]}"; do
for l2_variant_reg_strength in "${l2_variant_reg_strengths[@]}"; do
for variant_encoder_architecture in "${variant_encoder_architectures[@]}"; do
	model_training_output_stem=${model_training_dir}"full_rss_model_single_tissue_expr_var_sdev_"${test_tissue}"_lr_"${learning_rate}"_l2t_NA_l2v_"${l2_variant_reg_strength}"_var_arch_"${variant_encoder_architecture//,/x}
	sbatch borzoi_full_rss_single_tissue_expr_norm_model_training.sh $gtex_tissue_names_file $prediction_inv_ld_input_data_summary_filestem $test_tissue $model_training_output_stem $learning_rate $l2_variant_reg_strength $variant_encoder_architecture $gtex_tpm_expression $gtex_sample_attributes_file
done
done
done
fi







##########
# Evaluate standard borzoi predictions
if false; then
source ~/.bashrc
conda activate borzoi
python organize_predictions.py $borzoi_gtex_tissues_file ${model_training_dir}
fi



if false; then
source ~/.bashrc
conda activate plink_env 
Rscript visualize_predictions.R $borzoi_gtex_tissues_file ${model_training_dir} $visualization_dir
fi









