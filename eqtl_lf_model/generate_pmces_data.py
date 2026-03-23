import numpy as np
import os
import sys
import pdb




def create_mapping_from_vgt_to_pmces(fine_map_summary_file):
	f = open(fine_map_summary_file)
	dicti = {}
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			header = np.asarray(data)
			continue
		if data[9] != 'SUSIE':
			continue
		variant_id = data[4]
		gene_id = data[11].split('.')[0]
		tissue_name =data[10]
		beta_posterior = float(data[18])
		vgt = variant_id + ':' + gene_id + ':' + tissue_name

		dicti[vgt] = beta_posterior

	f.close()

	return dicti


####################
# Command line args
####################
pmces_eqtl_effect_size_file = sys.argv[1]
pmces_se_eqtl_effect_size_file = sys.argv[2]
eqtl_effect_size_file = sys.argv[3]
eqtl_se_file = sys.argv[4]
fine_map_summary_file = sys.argv[5]


# Mapping from variant-gene-tissue to pmces
vgt_to_pmces = create_mapping_from_vgt_to_pmces(fine_map_summary_file)




f = open(eqtl_effect_size_file)
t = open(pmces_eqtl_effect_size_file,'w')
t_se = open(pmces_se_eqtl_effect_size_file,'w')

head_count = 0
for line in f:
	line = line.rstrip()
	data = line.split('\t')
	if head_count == 0:
		head_count = head_count + 1
		t.write(line + '\n')
		t_se.write(line + '\n')
		tissue_names = np.asarray(data[1:])
		continue
	variant_id = data[0].split(':')[0]
	gene_id = data[0].split(':')[1].split('.')[0]
	marginal_effects = np.asarray(data[1:])

	tmp_arr = []
	tmp_arr2 = []
	booler = False
	for tiss_iter, tissue_name in enumerate(tissue_names):
		if marginal_effects[tiss_iter] == 'nan':
			tmp_arr.append('nan')
			tmp_arr2.append('nan')
		else:
			vgt = variant_id + ':' + gene_id + ':' + tissue_name
			if vgt in vgt_to_pmces:
				booler = True
				tmp_arr.append(str(vgt_to_pmces[vgt]))
			else:
				tmp_arr.append('0.0')
			tmp_arr2.append('0.01')
	tmp_arr = np.asarray(tmp_arr)
	tmp_arr2 = np.asarray(tmp_arr2)
	if booler == True:
		t.write(data[0] + '\t' + '\t'.join(tmp_arr) + '\n')
		t_se.write(data[0] + '\t' + '\t'.join(tmp_arr2) + '\n')

f.close()
t.close()
t_se.close()






