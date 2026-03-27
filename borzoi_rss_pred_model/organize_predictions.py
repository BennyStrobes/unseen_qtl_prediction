import numpy as np
import os
import sys
import pdb



def sem(arr):
    return np.std(arr, ddof=1) / np.sqrt(len(arr))


###################
borzoi_gtex_tissues_file = sys.argv[1]
model_training_dir = sys.argv[2]


tissue_names = np.loadtxt(borzoi_gtex_tissues_file,dtype=str,delimiter='\t')[1:,0]


output_file = model_training_dir + 'results_summary.txt'
t = open(output_file,'w')
t.write('tissue_name\tmethod\tmean\tmean_se\n')
for tissue_name in tissue_names:
	borzoi_eval_file = model_training_dir + 'borzoi_eval_' + tissue_name + '_all_gene_test_tissue_evaluation.txt'
	mod_pred_eval_file = model_training_dir + 'full_rss_model_held_out_genes_eval_train_test_tissue_' + tissue_name + '_lr_1e-5_l2t_100.0_l2v_100.0_var_arch_128x64x32_all_gene_test_tissue_evaluation.txt'

	if os.path.exists(mod_pred_eval_file) == False:
		continue

	borzoi_df = np.loadtxt(borzoi_eval_file,dtype=str,delimiter='\t')
	mod_pred_df = np.loadtxt(mod_pred_eval_file,dtype=str,delimiter='\t')

	indices = (borzoi_df[:,1] == 'test') & (borzoi_df[:,-1] != 'nan')
	indices = (borzoi_df[:,1] != 'gene_split') & (borzoi_df[:,-1] != 'nan')

	#indices = borzoi_df[:,-1] != 'nan'

	borzoi_pred_expr_corrs = borzoi_df[indices, -1].astype(float)
	mod_pred_expr_corrs = mod_pred_df[indices,-1].astype(float)

	
	t.write(tissue_name + '\t' + 'borzoi' + '\t' + str(np.mean(borzoi_pred_expr_corrs)) + '\t' + str(sem(borzoi_pred_expr_corrs)) + '\n')
	t.write(tissue_name + '\t' + 'factor_model' + '\t' + str(np.mean(mod_pred_expr_corrs)) + '\t' + str(sem(mod_pred_expr_corrs)) + '\n')

t.close()