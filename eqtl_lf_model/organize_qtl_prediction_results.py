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
t.write('tissue\tsample_size\tclosest_tissue\tdist_to_closest_tissue\tloss\tloss_se\tcorrelation\tcorrelation_se\tnn_loss\tnn_loss_se\tnn_correlation\tnn_correlation_se\trt_loss\trt_loss_se\trt_correlation\trt_correlation_se\tresid_var\tpred_resid_var\n')
for tissue_name in tissue_names:
	sample_size = tissue_name_to_sample_size[tissue_name]

	qtl_pred_summary_file = model_training_output_stem + '_' + tissue_name + '_het_var_multi_restart_nearest_tissues_test_loss_summary.txt'
	nn_pred_summary_file = model_training_output_stem + '_' + tissue_name + '_het_var_multi_restart_nearest_tissues_nearest_tissue_pred_test_loss_summary.txt'
	rt_pred_summary_file = model_training_output_stem + '_' + tissue_name + '_het_var_multi_restart_nearest_tissues_random_tissue_pred_test_loss_summary.txt'
	nearest_tissue_file = model_training_output_stem + '_' + tissue_name + '_het_var_multi_restart_nearest_tissues_nearest_tissue_summary.txt'

	qtl_pred_pred_file = model_training_output_stem + '_' + tissue_name + '_het_var_multi_restart_nearest_tissues_test_preds.txt'

	model_loss_values = np.loadtxt(qtl_pred_summary_file,dtype=str,delimiter='\t')[1,1:]
	nn_loss_values = np.loadtxt(nn_pred_summary_file,dtype=str,delimiter='\t')[1,1:]
	nn_fields = np.loadtxt(nearest_tissue_file,dtype=str,delimiter='\t')[1,1:]
	rt_fields = np.loadtxt(rt_pred_summary_file,dtype=str,delimiter='\t')[1,1:]

	qtl_pred_preds = np.loadtxt(qtl_pred_pred_file,dtype=str,delimiter='\t')[1:,:].astype(float)
	beta_hat = qtl_pred_preds[:,0]
	beta_hat_se = qtl_pred_preds[:,1]
	pred_beta = qtl_pred_preds[:,2]
	pred_beta_var = qtl_pred_preds[:,3]

	#resid_var = np.var(beta_hat - pred_beta,ddof=1)
	#pred_resid_var = np.mean(np.square(beta_hat_se) + pred_beta_var)

	resid_var = np.var(beta_hat - pred_beta,ddof=1) - np.mean(np.square(beta_hat_se))
	pred_resid_var = np.mean(pred_beta_var)


	t.write(tissue_name + '\t' + sample_size + '\t' + '\t'.join(nn_fields) + '\t' + '\t'.join(model_loss_values) + '\t' + '\t'.join(nn_loss_values) + '\t' + '\t'.join(rt_fields) + '\t' + str(resid_var) + '\t' + str(pred_resid_var) + '\n')

t.close()

print(organized_results_file)
