


#########################
# Input data
#########################
# GTEx v10 summary statistics
# Note GTEx variant ids are of format: chr{chrom}_{position}_{REF}_{ALT}_b38
gtex_summary_stats_dir="/lab-share/CHIP-Strober-e2/Public/GTEx/eqtl_sumstats/"

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



#########################
# Output data
#########################
# Output root directory
output_root_dir="/lab-share/CHIP-Strober-e2/Public/ben/unseen_qtl_prediction/marginal_predictions/"

# Output directory processed summary statistics
tissue_names_dir=${output_root_dir}"tissue_names/"

# Output directory processed summary statistics
processed_gtex_sumstats_dir=${output_root_dir}"processed_gtex_sumstats/"

# Output directory containing processed expression
processed_expression_dir=${output_root_dir}"processed_gene_expression/"




#########################
# Scripts
#########################

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
	tissue_name="Whole_Blood"
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



