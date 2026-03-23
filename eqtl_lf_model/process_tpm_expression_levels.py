import numpy as np
import os
import sys
import pdb
import re
import gzip

def to_underscore(s: str) -> str:
	s = s.replace(")", "")
	# replace " (" OR any run of spaces and hyphens with "_"
	return re.sub(r"(?:\s*\(|[ -]+)", "_", s)


def create_mapping_from_gtex_sample_id_to_individual_tissue_format(gtex_sample_attributes_file, gtex_tissue_names_arr):
	gtex_sample_to_tissue_name = {}
	gtex_tissues = {}
	f = open(gtex_sample_attributes_file)
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		gtex_sample_id = data[0]
		tissue_type = to_underscore(data[6])
		if tissue_type == 'Cells_EBV_transformed_lymphocytes':
			tissue_type = 'Cells_EBV-transformed_lymphocytes'
		elif tissue_type == 'Brain_Spinal_cord_cervical_c_1':
			tissue_type = 'Brain_Spinal_cord_cervical_c-1'


		individual_id = gtex_sample_id.split('-')[0] + '-' + gtex_sample_id.split('-')[1]
		gtex_sample_to_tissue_name[gtex_sample_id] = individual_id + ':' + tissue_type
		gtex_tissues[tissue_type] = 1
	f.close()


	for tissue in gtex_tissue_names_arr:
		if tissue not in gtex_tissues:
			print('assumption erororo')
			pdb.set_trace()



	return gtex_sample_to_tissue_name


def extract_gtex_tissue_names(gtex_tissue_names_file):
	f = open(gtex_tissue_names_file)
	arr = []
	dicti = {}
	head_count = 0
	for line in f:
		line = line.rstrip()
		if head_count == 0:
			head_count = head_count + 1
			continue
		arr.append(line)
		dicti[line] = 1

	f.close()
	return np.asarray(arr), dicti

def create_dictionary_list_of_samples_used_in_eqtl_analysis(gtex_tissue_names_arr, gtex_per_tissue_expression_dir):
	dicti = {}
	for tissue_name in gtex_tissue_names_arr:
		gzipped_expr_file = gtex_per_tissue_expression_dir + tissue_name + '.v10.normalized_expression.bed.gz'
		f = gzip.open(gzipped_expr_file)
		for line in f:
			line = line.decode('utf-8').rstrip()
			data = line.split('\t')
			individual_ids = np.asarray(data[4:])
			for individual_id in individual_ids:
				individual_tissue_name = individual_id + ':' + tissue_name
				if individual_tissue_name in dicti:
					print('assumtpinerorno')
					pdb.set_trace()
				dicti[individual_tissue_name] = 1
			break
		f.close()
	return dicti


def extract_dictionary_list_of_protein_coding_genes(gencode_gene_annotation_file):
	f = open(gencode_gene_annotation_file)
	gene_names = {}
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if len(data) != 9:
			print('assumption eroror')
			pdb.set_trace()
		ensamble_id = data[8].split(';')[0].split('"')[1]
		if ensamble_id.startswith('ENSG') == False:
			print('assumptione rororo')
			pdb.set_trace()
		gene_names[ensamble_id] = 1
	f.close()
	return gene_names


def filter_and_process_expression_levels(gtex_tpm_expression, gtex_sample_id_to_individual_tissue_format, eqtl_sample_list, protein_coding_genes, processed_all_sample_expression_file, processed_sample_names_output_file):
	skip1 = 0
	skip2 = 0
	f = gzip.open(gtex_tpm_expression)
	t = open(processed_all_sample_expression_file,'w')
	head_count = 0
	for line in f:
		line = line.decode('utf-8').rstrip()
		data = line.split('\t')
		# Irrelevent header lines
		if head_count < 2:
			head_count = head_count + 1
			continue
		# Header
		if head_count == 2:
			head_count = head_count + 1
			t.write(data[0] + '\t' + data[1] + '\t')
			sample_ids = np.asarray(data[2:])
			valid_sample_indices =[]
			filtered_individual_tissue_ids2 = []
			for sample_id in sample_ids:
				if sample_id not in gtex_sample_id_to_individual_tissue_format:
					valid_sample_indices.append(False)
				else:
					individual_tissue_name = gtex_sample_id_to_individual_tissue_format[sample_id]
					if individual_tissue_name in eqtl_sample_list:
						valid_sample_indices.append(True)
						filtered_individual_tissue_ids2.append(individual_tissue_name)
					else:
						valid_sample_indices.append(False)
			valid_sample_indices = np.asarray(valid_sample_indices)
			filtered_sample_ids = sample_ids[valid_sample_indices]
			filtered_individual_tissue_ids = []
			for sample_id in filtered_sample_ids:
				filtered_individual_tissue_ids.append(gtex_sample_id_to_individual_tissue_format[sample_id])
			filtered_individual_tissue_ids = np.asarray(filtered_individual_tissue_ids)
			filtered_individual_tissue_ids2 = np.asarray(filtered_individual_tissue_ids2)
			# Quick error checking
			if np.array_equal(filtered_individual_tissue_ids, filtered_individual_tissue_ids2) == False:
				print('assumption eoror')
				pdb.set_trace()
			t.write('\t'.join(filtered_individual_tissue_ids) + '\n')
			# Write sample names file
			t2 = open(processed_sample_names_output_file,'w')
			t2.write('gtex_sample_id\tgtex_individual_tissue_id\tindividual_id\ttissue_id\n')
			for ii, sample_id in enumerate(filtered_sample_ids):
				individual_tissue_id = filtered_individual_tissue_ids[ii]
				# QUick error check
				if individual_tissue_id != gtex_sample_id_to_individual_tissue_format[sample_id]:
					print('assumption eororor')
					pdb.set_trace()
				# Print to output
				t2.write(sample_id + '\t' + individual_tissue_id + '\t' + individual_tissue_id.split(':')[0] + '\t' + individual_tissue_id.split(':')[1] + '\n')
			t2.close()
			continue

		# Standard line
		# QUick error check
		if len(data) != 19618:
			print('assumptione rororo')
			pdb.set_trace()
		ensamble_id = data[0]
		# Only print protein coding genes
		if ensamble_id.startswith('ENSG') == False:
			print('error')
			pdb.set_trace()
		if ensamble_id not in protein_coding_genes:
			skip1 = skip1 + 1
			continue
		# Filter TPM measurements
		expr_levels = np.asarray(data[2:])
		filtered_expression_levels = expr_levels[valid_sample_indices]
		# Log transform
		log_filtered_expression_levels = np.log2(filtered_expression_levels.astype(float) + 1)
		# Filter out genes with no variance
		if np.var(log_filtered_expression_levels) == 0:
			skip2 = skip2 + 1
			continue

		# Print to output
		t.write(data[0] + '\t' + data[1] + '\t')
		t.write('\t'.join(log_filtered_expression_levels.astype(str)) + '\n')
	f.close()
	t.close()
	pdb.set_trace()
	return


####################
# Command line args
####################
gtex_tpm_expression = sys.argv[1]
gtex_tissue_names_file = sys.argv[2]
gtex_per_tissue_expression_dir = sys.argv[3]
gencode_gene_annotation_file = sys.argv[4]
gtex_sample_attributes_file = sys.argv[5]
processed_all_sample_expression_file = sys.argv[6] # Expression output file
processed_sample_names_output_file = sys.argv[7] 


# First Extract gtex tissue names
gtex_tissue_names_arr, gtex_tissue_names_dicti = extract_gtex_tissue_names(gtex_tissue_names_file)

# Second create dictionary mapping gtex full sample id to "individual_ID:tissue_name" format
gtex_sample_id_to_individual_tissue_format = create_mapping_from_gtex_sample_id_to_individual_tissue_format(gtex_sample_attributes_file, gtex_tissue_names_arr)

# Third extract dictionary list of all samples used for eQTL analysis (in "individual_ID:tissue_name" format)
eqtl_sample_list = create_dictionary_list_of_samples_used_in_eqtl_analysis(gtex_tissue_names_arr, gtex_per_tissue_expression_dir)

# Fourth extract dictionary list of ensamble ids corresponding to protein coding genes
protein_coding_genes = extract_dictionary_list_of_protein_coding_genes(gencode_gene_annotation_file)

# Fifth, filter TPM expression file to those in eqtl_sample_list and filter genes to those in protein_coding_genes
# Also change from TPM To log(TPM + 1)
filter_and_process_expression_levels(gtex_tpm_expression, gtex_sample_id_to_individual_tissue_format, eqtl_sample_list, protein_coding_genes, processed_all_sample_expression_file, processed_sample_names_output_file)








