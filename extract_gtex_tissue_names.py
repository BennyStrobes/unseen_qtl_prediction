import numpy as np
import os
import sys
import pdb






######################
# Command line args
######################
gtex_tissue_names_file = sys.argv[1] # output file
input_ss_dir = sys.argv[2]


##############
# Extract tissue names
tissue_names = []
for file_name in os.listdir(input_ss_dir):
	if file_name.endswith('chr1.parquet') == False:
		continue
	tissue_name = file_name.split('.v10.all')[0]
	tissue_names.append(tissue_name)
tissue_names = np.asarray(tissue_names)
tissue_names = np.sort(tissue_names)
if len(tissue_names) != 50:
	print('assumption erororo')
	pdb.set_trace()



##############
# Print to output
t = open(gtex_tissue_names_file, 'w')
t.write('tissue_name\n')
for tissue_name in tissue_names:
	t.write(tissue_name + '\n')
t.close()
