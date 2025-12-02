import numpy as np
import os
import sys
import pdb



def extract_tissue_names(gtex_tissue_names_file):
	arr = []
	f = open(gtex_tissue_names_file)
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		arr.append(line)
	f.close()

	arr = np.asarray(arr)

	# Quick error check
	if len(arr) != len(np.unique(arr)):
		print('assumption erororor')
		pdb.set_trace()

	return arr

def print_to_output(test_tissues, test_tissue_file):
	t = open(test_tissue_file,'w')
	t.write('tissue_name\n')
	for tissue in test_tissues:
		t.write(tissue + '\n')
	t.close()
	return



####################
# Command line args
####################
gtex_tissue_names_file = sys.argv[1]
n_test_tissues = int(sys.argv[2])
training_tissue_file = sys.argv[3]  # Output file
test_tissue_file = sys.argv[4]  # Output file
seed = int(sys.argv[5])

# Set random seed
np.random.seed(seed)


# Extract ordered list of tissue names
tissue_names = extract_tissue_names(gtex_tissue_names_file)

# Get test tissues
test_tissues = np.sort(np.random.choice(tissue_names,size=n_test_tissues, replace=False))

# Get train tissues
train_tissues = []
for tissue in tissue_names:
	if tissue not in test_tissues:
		train_tissues.append(tissue)
train_tissues = np.asarray(train_tissues)

print_to_output(train_tissues, training_tissue_file)
print_to_output(test_tissues, test_tissue_file)

