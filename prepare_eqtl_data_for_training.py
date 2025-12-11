import numpy as np
import os
import sys
import pdb




def get_original_training_tissues(train_summary_file):
	f = open(train_summary_file)
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		tissues = []
		for ele in data[1:]:
			tissues.append(ele.split(':')[1])
		tissues = np.asarray(tissues)
		break

	f.close()
	return tissues


def extract_list_of_variant_gene_pairs_across_all_tissues_on_chrom(original_training_tissues, processed_gtex_sumstats_dir, chrom_string):
	dicti = {}
	for training_tissue in original_training_tissues:
		eqtl_ss_file = processed_gtex_sumstats_dir + training_tissue + '_chrom' + chrom_string + '_150000_hm3_eqtl_summary_stats.txt'
		f = open(eqtl_ss_file)
		head_count = 0
		for line in f:
			line = line.rstrip()
			data = line.split('\t')
			if head_count == 0:
				head_count = head_count + 1
				continue
			variant_gene_pair = data[1] + ':' + data[0]
			pos = int(data[1].split('_')[1])
			if variant_gene_pair not in dicti:
				dicti[variant_gene_pair] = pos
		f.close()
	sorted_variant_gene_pairs =  sorted(dicti.keys(), key=dicti.get)
	return sorted_variant_gene_pairs

def print_summary_stats(t_beta, t_se, variant_gene_pairs, original_tissues, processed_gtex_sumstats_dir, chrom_string):
	dicti_beta = {}
	dicti_se = {}
	TT = len(original_tissues)
	for variant_gene_pair in variant_gene_pairs:
		dicti_beta[variant_gene_pair] = np.full(TT, np.nan)
		dicti_se[variant_gene_pair] = np.full(TT, np.nan)
	

	for tt, tissue in enumerate(original_tissues):
		eqtl_ss_file = processed_gtex_sumstats_dir + tissue + '_chrom' + chrom_string + '_150000_hm3_eqtl_summary_stats.txt'
		f = open(eqtl_ss_file)
		head_count = 0
		for line in f:
			line = line.rstrip()
			data = line.split('\t')
			if head_count == 0:
				head_count = head_count + 1
				continue
			variant_gene_pair = data[1] + ':' + data[0]

			# Extract effect sizes and standard errors
			beta_hat = float(data[7])
			beta_hat_se = float(data[8])
			af = float(data[3])
			genotype_sdev = np.sqrt(2*(af)*(1.0-af))

			# Convert effect sizes and standard errors to standardized genotype version
			std_beta_hat = beta_hat*genotype_sdev
			std_beta_hat_se = beta_hat_se*genotype_sdev

			dicti_beta[variant_gene_pair][tt] = std_beta_hat
			dicti_se[variant_gene_pair][tt] = std_beta_hat_se
		f.close()

	# Print to output
	for variant_gene_pair in variant_gene_pairs:
		t_beta.write(variant_gene_pair + '\t' + '\t'.join(dicti_beta[variant_gene_pair].astype(str)) + '\n')
		t_se.write(variant_gene_pair + '\t' + '\t'.join(dicti_se[variant_gene_pair].astype(str)) + '\n')

	return t_beta, t_se

###############
# Command line args
###############

eqtl_effect_size_file = sys.argv[1]
eqtl_se_file = sys.argv[2]
single_samp_per_tissue_pc_file = sys.argv[3]
processed_gtex_sumstats_dir = sys.argv[4]


# Get original training tissues
original_tissues = get_original_training_tissues(single_samp_per_tissue_pc_file)



# Open output file handles
t_train_beta = open(eqtl_effect_size_file,'w')
t_train_se = open(eqtl_se_file,'w')


# Add header
t_train_beta.write('beta_hat\t' + '\t'.join(original_tissues) + '\n')
t_train_se.write('beta_hat_se\t' + '\t'.join(original_tissues) + '\n')


for chrom_num in range(1,23):
	print(chrom_num)
	# Extract ordered array of variant gene pairs found in at least 1 tissue
	variant_gene_pairs = extract_list_of_variant_gene_pairs_across_all_tissues_on_chrom(original_tissues, processed_gtex_sumstats_dir, str(chrom_num))
	
	# Print summary stats
	# for training
	t_train_beta, t_train_se = print_summary_stats(t_train_beta, t_train_se, variant_gene_pairs, original_tissues, processed_gtex_sumstats_dir, str(chrom_num))


t_train_beta.close()
t_train_se.close()

