import numpy as np
import os
import sys
import pdb


def extract_tissue_names(training_tissue_file):
	f = open(training_tissue_file)
	head_count = 0
	arr = []
	for line in f:
		line = line.rstrip()
		if head_count == 0:
			head_count = head_count + 1
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
tissue_file = sys.argv[1]
seed = int(sys.argv[2])
expression_file = sys.argv[3]
expression_output_file = sys.argv[4]

np.random.seed(seed)

ordered_tissue_names = extract_tissue_names(tissue_file)


f = open(expression_file)
t = open(expression_output_file,'w')
head_count = 0
for line in f:
	line = line.rstrip()
	data = line.split('\t')
	if head_count == 0:
		head_count = head_count + 1
		sample_names = np.asarray(data[2:])

		# Randomly select one sample per tissue
		new_indices = []
		for tissue_name in ordered_tissue_names:
			idx = np.where(np.char.find(sample_names, tissue_name) != -1)[0]
			# Randomly select one for this tissue
			selected_index = np.random.choice(idx)
			new_indices.append(selected_index)
		new_indices = np.asarray(new_indices)

		filtered_sample_names = sample_names[new_indices]
		t.write(data[0] + '\t' + '\t'.join(filtered_sample_names) + '\n')
		continue
	sample_names = np.asarray(data[2:])
	filtered_sample_names = sample_names[new_indices]
	t.write(data[0] + '\t' + '\t'.join(filtered_sample_names) + '\n')

f.close()
t.close()