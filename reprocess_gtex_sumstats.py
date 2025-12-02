import numpy as np
import os
import sys
import pdb
import pyarrow.parquet as pq
import pandas as pd
import time
import gzip

def extract_dictionary_list_of_variants_that_pass_filters(snp_set, chrom_num, genotype_dir_1000_G):
	# First get hm3 rsids
	hm3_rsids = {}
	hm3_rsid_file = genotype_dir_1000_G + 'w_hm3.noMHC.snplist'
	f = open(hm3_rsid_file)
	for line in f:
		line = line.rstrip()
		hm3_rsids[line] = 1
	f.close()

	# Now get GTEx format ids
	gtex_ids = {}
	bim_file = genotype_dir_1000_G + '1000G.EUR.hg38.' + str(chrom_num) + '.bim'
	f = open(bim_file)
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		rsid = data[1]
		if rsid not in hm3_rsids and snp_set == 'hm3':
			continue
		gtex_var_id1 = 'chr' + str(chrom_num) + '_' + data[3] + '_' + data[4] + '_' + data[5] + '_b38'
		gtex_var_id2 = 'chr' + str(chrom_num) + '_' + data[3] + '_' + data[5] + '_' + data[4] + '_b38'

		gtex_ids[gtex_var_id1] = 1
		gtex_ids[gtex_var_id2] = 1
	f.close()
	return gtex_ids


def extract_per_tissue_sample_size(file_name):
	f = gzip.open(file_name)
	for line in f:
		line = line.decode('utf-8').rstrip()
		data = line.split('\t')
		gtex_sample_size = len(data[4:])
		break
	f.close()
	return gtex_sample_size

#####################
# Command line args
#####################
tissue_name = sys.argv[1]
processed_gtex_sumstats_dir = sys.argv[2]
gtex_summary_stats_dir = sys.argv[3]
genotype_dir_1000_G = sys.argv[4]
cis_window = sys.argv[5]
snp_set = sys.argv[6]
gtex_per_tissue_expression_dir = sys.argv[7]


# Extract expression sample size
samp_size = extract_per_tissue_sample_size(gtex_per_tissue_expression_dir + tissue_name + '.v10.normalized_expression.bed.gz')
samp_size_string = str(samp_size)

cis_window_float = float(cis_window)

# Loop through autosomal chromosomes
for chrom_num in range(1,23):
	print(chrom_num)
	# Extract dictionary list of variants that pass filters on this chromosome
	variant_dicti = extract_dictionary_list_of_variants_that_pass_filters(snp_set, chrom_num, genotype_dir_1000_G)

	# Open output file
	output_file = processed_gtex_sumstats_dir + tissue_name + '_chrom' + str(chrom_num) + '_' + cis_window + '_' + snp_set + '_eqtl_summary_stats.txt'
	t = open(output_file,'w')

	# Loop through parquet file containing summary stats for this chromosome
	chrom_parquet_file = gtex_summary_stats_dir + tissue_name + '.v10.allpairs.chr' + str(chrom_num) + '.parquet'
	pf = pq.ParquetFile(chrom_parquet_file)
	counter = 0
	for rg in range(pf.num_row_groups):
		table = pf.read_row_group(rg)   # this is a chunk

		# Print header if first loop
		if counter == 0:
			df = table.to_pandas()       
			column_names = np.asarray(df.columns)
			t.write('\t'.join(column_names) + '\t' + 'sample_size' + '\n')
			counter = counter + 1

		# Process fields and print to output
		array = np.asarray(table)

		# Loop through snp-gene pairs 
		nrows = array.shape[0]
		for row_iter in range(nrows):
			data = array[row_iter, :]
			variant_id = data[1]
			abs_distance_to_tss = abs(float(data[2]))

			# Ignore non-hm3 snps
			if variant_id not in variant_dicti:
				continue
			# Ignore variant-gene pairs where variant is too far away from gene
			if abs_distance_to_tss > cis_window_float:
				continue

			# Print to output
			t.write('\t'.join(data.astype(str)) + '\t' + str(samp_size_string) + '\n')
	t.close()
