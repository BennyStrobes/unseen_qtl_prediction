import numpy as np
import os
import sys
import pdb
import re

def to_underscore(s: str) -> str:
	s = s.replace(")", "")
	# replace " (" OR any run of spaces and hyphens with "_"
	return re.sub(r"(?:\s*\(|[ -]+)", "_", s)

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

def extract_borzoi_target_names(borzoi_target_file):
	f = open(borzoi_target_file)
	head_count = 0
	target_identifiers = []
	target_descriptions = []
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		identifier = data[1]
		if identifier.endswith('+'):
			identifier = identifier.split('+')[0]
		elif identifier.endswith('-'):
			continue
		description = data[-1]
		target_identifiers.append(identifier)
		target_descriptions.append(description)
	f.close()

	return np.asarray(target_identifiers), np.asarray(target_descriptions)


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

def create_mapping_from_gtex_tissue_to_target_indices(target_summary_file):
	f = open(target_summary_file)
	dicti = {}
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count= head_count + 1
			continue
		if data[-1] == 'NA':
			continue
		tissue_name = data[-1]
		if tissue_name not in dicti:
			dicti[tissue_name] = []
		dicti[tissue_name].append(int(data[0]))
	f.close()
	return dicti

######################
# Command line args
######################
borzoi_target_file = sys.argv[1]
gtex_tissue_names_file = sys.argv[2]
gtex_sample_attributes_file = sys.argv[3]
target_summary_file = sys.argv[4]
borzoi_gtex_tissues_file = sys.argv[5]

borzoi_target_identifiers, borzoi_target_descriptions = extract_borzoi_target_names(borzoi_target_file)


# First Extract gtex tissue names
gtex_tissue_names_arr, gtex_tissue_names_dicti = extract_gtex_tissue_names(gtex_tissue_names_file)

# Second create dictionary mapping gtex full sample id to "individual_ID:tissue_name" format
gtex_sample_id_to_individual_tissue_format = create_mapping_from_gtex_sample_id_to_individual_tissue_format(gtex_sample_attributes_file, gtex_tissue_names_arr)

t = open(target_summary_file,'w')
t.write('target_index\ttarget_identifier\ttarget_description\tGTEx_tissue\n')
for ii, identifier in enumerate(borzoi_target_identifiers):

	if identifier.split('.')[0] not in gtex_sample_id_to_individual_tissue_format:
		t.write(str(ii) + '\t' + identifier + '\t' + borzoi_target_descriptions[ii] + '\t' + 'NA' + '\n')
	else:
		ind_tissue_name = gtex_sample_id_to_individual_tissue_format[identifier.split('.')[0]]
		tissue_name = ind_tissue_name.split(':')[1]
		t.write(str(ii) + '\t' + identifier + '\t' + borzoi_target_descriptions[ii] + '\t' + tissue_name + '\n')
t.close()

# create mapping from GTEx tissue to target index
gtex_tissue_to_target_indices = create_mapping_from_gtex_tissue_to_target_indices(target_summary_file)


t = open(borzoi_gtex_tissues_file,'w')
t.write('tissue_name\tborzoi_target_index\n')
for tissue_name in gtex_tissue_names_arr:
	if tissue_name not in gtex_tissue_to_target_indices:
		continue
	tissue_indices = gtex_tissue_to_target_indices[tissue_name]
	tmp_index = tissue_indices[0]
	t.write(tissue_name + '\t' + str(tmp_index) + '\n')
t.close()

print(borzoi_gtex_tissues_file)




