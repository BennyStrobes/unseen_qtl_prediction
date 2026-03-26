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

# Ordered gtex tissues names 
gtex_tissue_names_file="/lab-share/CHIP-Strober-e2/Public/ben/s2e_uncertainty/eqtl_borzoi_integration_processed_data/borzoi/ordered_gtex_tissues_chr1.txt"


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
# Fit model 
test_tissue="Adipose_Visceral_Omentum"
learning_rates=("1e-5" "1e-4" "3e-4" "1e-3")
l2_tissue_reg_strengths=("1e-3" "1.0" "100.0" "1000.0")
l2_variant_reg_strengths=("1e-3" "1.0" "100.0" "1000.0")
variant_encoder_architecture="128,64,32"
if false; then
for learning_rate in "${learning_rates[@]}"; do
for l2_tissue_reg_strength in "${l2_tissue_reg_strengths[@]}"; do
for l2_variant_reg_strength in "${l2_variant_reg_strengths[@]}"; do

	model_training_output_stem=${model_training_dir}"model_held_out_genes3_eval_train_test_tissue_"${test_tissue}"_lr_"${learning_rate}"_l2t_"${l2_tissue_reg_strength}"_l2v_"${l2_variant_reg_strength}"_var_arch_"${variant_encoder_architecture//,/x}
	sbatch borzoi_rss_model_training.sh $gtex_tissue_names_file $single_samp_per_tissue_expr_file $prediction_input_data_summary_filestem $test_tissue $model_training_output_stem $learning_rate $l2_tissue_reg_strength $l2_variant_reg_strength $variant_encoder_architecture
done
done
done
fi

learning_rate="1e-4"
l2_tissue_reg_strength="100.0"
l2_variant_reg_strength="100.0"
	model_training_output_stem=${model_training_dir}"model_held_out_genes4_eval_train_test_tissue_"${test_tissue}"_lr_"${learning_rate}"_l2t_"${l2_tissue_reg_strength}"_l2v_"${l2_variant_reg_strength}"_var_arch_"${variant_encoder_architecture//,/x}
	if false; then
	sh borzoi_rss_model_training.sh $gtex_tissue_names_file $single_samp_per_tissue_expr_file $prediction_input_data_summary_filestem $test_tissue $model_training_output_stem $learning_rate $l2_tissue_reg_strength $l2_variant_reg_strength $variant_encoder_architecture
fi
if false; then
test_tissue="Heart_Left_Ventricle"
fi


learning_rate="1e-4"
l2_tissue_reg_strength="100.0"
l2_variant_reg_strength="100.0"
if false; then
tail -n +2 $gtex_tissue_names_file | while read -r test_tissue; do
	model_training_output_stem=${model_training_dir}"model_held_out_genes4_eval_train_test_tissue_"${test_tissue}"_lr_"${learning_rate}"_l2t_"${l2_tissue_reg_strength}"_l2v_"${l2_variant_reg_strength}"_var_arch_"${variant_encoder_architecture//,/x}
	sbatch borzoi_rss_model_training.sh $gtex_tissue_names_file $single_samp_per_tissue_expr_file $prediction_input_data_summary_filestem $test_tissue $model_training_output_stem $learning_rate $l2_tissue_reg_strength $l2_variant_reg_strength $variant_encoder_architecture
done
fi

test_tissue="Adipose_Visceral_Omentum"
learning_rate="1e-3"
l2_tissue_reg_strength="0.0"
l2_variant_reg_strength="0.0"
model_training_output_stem=${model_training_dir}"cat_model_held_out_genes4_eval_train_test_tissue_"${test_tissue}"_lr_"${learning_rate}"_l2t_"${l2_tissue_reg_strength}"_l2v_"${l2_variant_reg_strength}"_var_arch_"${variant_encoder_architecture//,/x}
sh borzoi_rss_model_training_batched.sh $gtex_tissue_names_file $single_samp_per_tissue_expr_file $prediction_input_data_summary_filestem $test_tissue $model_training_output_stem $learning_rate $l2_tissue_reg_strength $l2_variant_reg_strength $variant_encoder_architecture





























##################
# OLD (No longer used)
##################

##########
# Extract names of gtex tissues
gtex_tissue_names_file=${tissue_names_dir}"gtex_tissue_names.txt"
if false; then
source ~/.bashrc
conda activate borzoi
python extract_gtex_tissue_names.py $gtex_tissue_names_file $gtex_summary_stats_dir
fi



##########
# Re-process gtex summary statistics
cis_window="150000"
snp_set="hm3" # Either all or hm3
# Loop through tissues
if false; then
tail -n +2 "$gtex_tissue_names_file" | while IFS= read -r tissue_name; do
	sbatch process_gtex_sumstats.sh $tissue_name $processed_gtex_sumstats_dir $gtex_summary_stats_dir $genotype_dir_1000_G $cis_window $snp_set $gtex_per_tissue_expression_dir
done
fi


##########
# Process gene expression levels
processed_all_sample_expression_file=${processed_expression_dir}"log_transformed_tpm_expression_all_samples.txt"
processed_all_sample_sample_names_file=${processed_expression_dir}"log_transformed_tpm_expression_all_sample_names.txt"
if false; then
sh process_tpm_expression_levels.sh $gtex_tpm_expression $gtex_tissue_names_file $gtex_per_tissue_expression_dir $gencode_gene_annotation_file $gtex_sample_attributes_file $processed_all_sample_expression_file $processed_all_sample_sample_names_file
fi


##########
# Split expression between training and test data
# Also generate PCs for training and test data
n_pcs="20"
expression_pc_file=${processed_expression_dir}"log_transformed_tpm_expression_PC_nPCs_"${n_pcs}".txt"
if false; then
source ~/.bashrc
conda activate borzoi
python get_expression_pcs.py $processed_all_sample_expression_file $expression_pc_file $n_pcs
fi



##########
# Randomly select single training sample and single test sample for model fitting
seed="1"
single_samp_per_tissue_pc_file=${training_data_dir}"single_sample_per_tissue_expression_pcs_nPCs_"${n_pcs}"_seed"${seed}".txt"
single_samp_per_tissue_expr_file=${training_data_dir}"single_sample_per_tissue_expression_nPCs_"${n_pcs}"_seed"${seed}".txt"
if false; then
source ~/.bashrc
conda activate borzoi
python randomly_select_single_sample_for_each_tissue.py $expression_pc_file $gtex_tissue_names_file $single_samp_per_tissue_pc_file $seed $processed_all_sample_expression_file $single_samp_per_tissue_expr_file $fine_map_summary_file
fi



##########
# Prepare training input data
# TODO: SHOULD SCALE UP TO CHROMOSOMES BEYOND CHR1
eqtl_effect_size_file=${training_data_dir}"eqtl_effect_sizes_full_nPCs_"${n_pcs}"_seed"${seed}".txt"
eqtl_effect_size_file=${training_data_dir}"eqtl_effect_sizes_nPCs_"${n_pcs}"_seed"${seed}".txt"

eqtl_se_file=${training_data_dir}"eqtl_se_full_nPCs_"${n_pcs}"_seed"${seed}".txt"
eqtl_se_file=${training_data_dir}"eqtl_se_nPCs_"${n_pcs}"_seed"${seed}".txt"
if false; then
source ~/.bashrc
conda activate borzoi
python prepare_eqtl_data_for_training.py $eqtl_effect_size_file $eqtl_se_file $single_samp_per_tissue_pc_file $processed_gtex_sumstats_dir
fi

pmces_eqtl_effect_size_file=${training_data_dir}"pmces_eqtl_effect_sizes_nPCs_"${n_pcs}"_seed"${seed}".txt"
pmces_se_eqtl_effect_size_file=${training_data_dir}"pmces_eqtl_se_nPCs_"${n_pcs}"_seed"${seed}".txt"

if false; then
source ~/.bashrc
conda activate borzoi
python generate_pmces_data.py $pmces_eqtl_effect_size_file $pmces_se_eqtl_effect_size_file $eqtl_effect_size_file $eqtl_se_file $fine_map_summary_file
fi


gtex_tissue_names_file2="/lab-share/CHIP-Strober-e2/Public/ben/unseen_qtl_prediction/marginal_predictions/tissue_names/gtex_tissue_names2.txt"

##########
# Run inference
if false; then
tail -n +2 "$gtex_tissue_names_file2" | while IFS= read -r test_tissue; do
	output_stem=${model_training_dir}"expression_reduced_eqtls_nPCs_"${n_pcs}"_seed"${seed}"_test_tissue_"${test_tissue}"_het_var_multi_restart_nearest_tissues"
	sbatch run_eqtl_expression_factorization_inference.sh $eqtl_effect_size_file $eqtl_se_file $single_samp_per_tissue_expr_file $test_tissue $gtex_tissue_names_file2 $output_stem $single_samp_per_tissue_pc_file
done
fi



gtex_tissue_names_file2="/lab-share/CHIP-Strober-e2/Public/ben/unseen_qtl_prediction/marginal_predictions/tissue_names/gtex_tissue_names2.txt"
if false; then
test_tissue="Thyroid"
output_stem=${model_training_dir}"expression_reduced_eqtls_nPCs_"${n_pcs}"_seed"${seed}"_test_tissue_"${test_tissue}"_het_var_multi_restart_nearest_tissues"
sh run_eqtl_expression_factorization_inference.sh $eqtl_effect_size_file $eqtl_se_file $single_samp_per_tissue_expr_file $test_tissue $gtex_tissue_names_file2 $output_stem $single_samp_per_tissue_pc_file
fi




gtex_tissue_names_file2="/lab-share/CHIP-Strober-e2/Public/ben/unseen_qtl_prediction/marginal_predictions/tissue_names/gtex_tissue_names2.txt"

test_tissue="Skin_Not_Sun_Exposed_Suprapubic"
output_stem=${model_training_dir}"pmces_expression_reduced_eqtls_nPCs_"${n_pcs}"_seed"${seed}"_test_tissue_"${test_tissue}"_het_var"
if false; then
sh run_eqtl_expression_factorization_inference.sh $pmces_eqtl_effect_size_file $pmces_se_eqtl_effect_size_file $single_samp_per_tissue_expr_file $test_tissue $gtex_tissue_names_file2 $output_stem $single_samp_per_tissue_pc_file
fi



model_training_output_stem=${model_training_dir}"expression_reduced_eqtls_nPCs_"${n_pcs}"_seed"${seed}"_test_tissue"
organized_results_file=${model_training_dir}"expression_reduced_eqtls_nPCs_"${n_pcs}"_seed"${seed}"_het_var_multi_restart_nearest_tissues_organized_test_results.txt"
if false; then
source ~/.bashrc
conda activate borzoi
python organize_qtl_prediction_results.py $model_training_output_stem $gtex_tissue_names_file2 $processed_gtex_sumstats_dir $organized_results_file
fi

organized_results_file=${model_training_dir}"expression_reduced_eqtls_nPCs_"${n_pcs}"_seed"${seed}"_organized_test_results.txt"
organized_results_file2=${model_training_dir}"expression_reduced_eqtls_nPCs_"${n_pcs}"_seed"${seed}"_het_var_organized_test_results.txt"

if false; then
source ~/.bashrc
conda activate plink_env
Rscript visualize_results.R $organized_results_file $organized_results_file2 $visualization_dir
fi




##################
# OLDER
##################
##########
# Randomly select KK test tissues
n_test_tissues="5"
seed="50"
training_tissue_file=${tissue_names_dir}"gtex_training_tissue_names_seed_"${seed}".txt"
test_tissue_file=${tissue_names_dir}"gtex_test_tissue_names_seed_"${seed}".txt"
if false; then
source ~/.bashrc
conda activate borzoi
python randomly_select_training_and_test_tissues.py $gtex_tissue_names_file $n_test_tissues $training_tissue_file $test_tissue_file $seed
fi




##########
# Split expression between training and test data
# Also generate PCs for training and test data
n_pcs="20"
training_expression_file=${processed_expression_dir}"log_transformed_tpm_expression_training_"${seed}".txt"
test_expression_file=${processed_expression_dir}"log_transformed_tpm_expression_test_"${seed}".txt"
training_expression_pc_file=${processed_expression_dir}"log_transformed_tpm_expression_PC_training_nPCs_"${n_pcs}"_seed"${seed}".txt"
test_expression_pc_file=${processed_expression_dir}"log_transformed_tpm_expression_PC_test_nPCs_"${n_pcs}"_seed"${seed}".txt"
if false; then
source ~/.bashrc
conda activate borzoi
python get_training_and_test_expression_data_and_pcs.py $processed_all_sample_expression_file $training_tissue_file $test_tissue_file $training_expression_file $test_expression_file $training_expression_pc_file $test_expression_pc_file $n_pcs
fi


