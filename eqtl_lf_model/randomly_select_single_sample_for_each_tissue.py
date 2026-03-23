import numpy as np
import os
import sys
import pdb


def extract_tissue_names(training_tissue_file, valid_tissues):
	f = open(training_tissue_file)
	head_count = 0
	arr = []
	for line in f:
		line = line.rstrip()
		if head_count == 0:
			head_count = head_count + 1
			continue
		if line not in valid_tissues:
			continue
		arr.append(line)
	f.close()
	return np.asarray(arr)


def extract_valid_tissues(fine_map):
	f = open(fine_map)
	dicti = {}
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		dicti[data[10]] = 1
	f.close()
	return dicti

#####################
# Command line args
#####################
expression_pc_file = sys.argv[1]
tissue_file = sys.argv[2]
output_file = sys.argv[3]
seed = int(sys.argv[4])
expression_file = sys.argv[5]
expression_output_file = sys.argv[6]
fine_map = sys.argv[7]

valid_tissues = extract_valid_tissues(fine_map)

np.random.seed(seed)

ordered_tissue_names = extract_tissue_names(tissue_file, valid_tissues)


# Load in existing expression pc file
old_expr_pc_raw = np.loadtxt(expression_pc_file, dtype=str,delimiter='\t')
header_column = old_expr_pc_raw[:,0]
old_expr_pc = old_expr_pc_raw[:,1:]



new_indices = []
new_pc_cols = []
new_pc_cols.append(header_column)
pc_filtered_sample_names = []
for tissue_name in ordered_tissue_names:
	idx = np.where(np.char.find(old_expr_pc[0, :], tissue_name) != -1)[0]

	# Randomly select one
	selected_index = np.random.choice(idx)

	new_pc_cols.append(old_expr_pc[:,selected_index])
	new_indices.append(selected_index)
	pc_filtered_sample_names.append(old_expr_pc[0,selected_index])
new_indices = np.asarray(new_indices)
new_pc_mat = np.transpose(np.asarray(new_pc_cols))
pc_filtered_sample_names = np.asarray(pc_filtered_sample_names)

np.savetxt(output_file, new_pc_mat, fmt="%s", delimiter='\t')

f = open(expression_file)
t = open(expression_output_file,'w')
head_count = 0
for line in f:
	line = line.rstrip()
	data = line.split('\t')
	if head_count == 0:
		head_count = head_count + 1
		sample_names = np.asarray(data[2:])
		filtered_sample_names = sample_names[new_indices]
		t.write(data[0] + '\t' + '\t'.join(filtered_sample_names) + '\n')
		if np.array_equal(filtered_sample_names, pc_filtered_sample_names) == False:
			print('assumption eroror')
			pdb.set_trace()
		continue
	sample_names = np.asarray(data[2:])
	filtered_sample_names = sample_names[new_indices]
	t.write(data[0] + '\t' + '\t'.join(filtered_sample_names) + '\n')

f.close()
t.close()