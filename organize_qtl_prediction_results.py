import numpy as np
import os
import sys
import pdb

def extract_tissue_names(gtex_tissue_names_file):
	f = open(gtex_tissue_names_file)
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



def create_mapping_from_tissue_name_to_sample_size(tissue_names, gtex_summary_stats_dir):
	mapping = {}
	for tissue_name in tissue_names:
		filer = gtex_summary_stats_dir + tissue_name + '_chrom2_150000_hm3_eqtl_summary_stats.txt'
		f = open(filer)
		head_count = 0
		for line in f:
			line = line.rstrip()
			data = line.split('\t')
			if head_count == 0:
				head_count = head_count + 1
				continue
			sample_size = data[-1]
			mapping[tissue_name] = sample_size
			break
		f.close()

	return mapping


######################
# Command line args
######################
model_training_output_stem = sys.argv[1]
gtex_tissue_names_file = sys.argv[2]
gtex_summary_stats_dir = sys.argv[3]
organized_results_file = sys.argv[4]


# Extract tissue names
tissue_names = extract_tissue_names(gtex_tissue_names_file)

# Create mapping from tissue name to sample size
tissue_name_to_sample_size = create_mapping_from_tissue_name_to_sample_size(tissue_names, gtex_summary_stats_dir)

# Initialize output file
t = open(organized_results_file,'w')
# Header
t.write('tissue\tsample_size\tclosest_tissue\tdist_to_closest_tissue\tloss\tloss_se\tcorrelation\tcorrelation_se\tnn_loss\tnn_loss_se\tnn_correlation\tnn_correlation_se\n')
for tissue_name in tissue_names:
	sample_size = tissue_name_to_sample_size[tissue_name]

	qtl_pred_summary_file = model_training_output_stem + '_' + tissue_name + '_test_loss_summary.txt'
	nn_pred_summary_file = model_training_output_stem + '_' + tissue_name + '_nearest_tissue_pred_test_loss_summary.txt'
	nearest_tissue_file = model_training_output_stem + '_' + tissue_name + '_nearest_tissue_summary.txt'

	model_loss_values = np.loadtxt(qtl_pred_summary_file,dtype=str,delimiter='\t')[1,1:]
	nn_loss_values = np.loadtxt(nn_pred_summary_file,dtype=str,delimiter='\t')[1,1:]
	nn_fields = np.loadtxt(nearest_tissue_file,dtype=str,delimiter='\t')[1,1:]
	t.write(tissue_name + '\t' + sample_size + '\t' + '\t'.join(nn_fields) + '\t' + '\t'.join(model_loss_values) + '\t' + '\t'.join(nn_loss_values) + '\n')

t.close()
