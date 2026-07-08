import numpy as np
import os
import sys
import glob
import re
import pdb


def parse_params_from_log_file(log_file, test_tissue):
	# Log files are named:
	# full_rss_lf_model_train_test_tissue_{test_tissue}_lr_{lr}_l2t_{l2}_l1v_{l1}_var_arch_K_{K}_training_log.txt
	basename = os.path.basename(log_file)
	pattern = 'full_rss_lf_model_train_test_tissue_' + test_tissue + \
		r'_lr_(?P<lr>.+)_l2t_(?P<l2t>.+)_l1v_(?P<l1v>.+)_var_arch_K_(?P<n_factors>.+)_training_log\.txt'
	match = re.match(pattern, basename)
	if match is None:
		return None
	return match.groupdict()


def extract_best_epoch(log_file):
	# Returns (best_epoch, train_loss, val_loss, val_corr) for the epoch with the
	# best (lowest) validation loss, or None if the log has no completed epochs.
	f = open(log_file)
	head_count = 0
	best_row = None
	best_val_loss = np.inf
	for line in f:
		line = line.rstrip()
		if head_count == 0:
			head_count = head_count + 1
			continue
		data = line.split('\t')
		# epoch	train_loss	val_loss	val_corr	best
		if len(data) != 5:
			continue
		epoch = data[0]
		train_loss = float(data[1])
		val_loss = float(data[2])
		val_corr = float(data[3])
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_row = (epoch, train_loss, val_loss, val_corr)
	f.close()
	return best_row


#######################
# Command line args
#######################
test_tissue = sys.argv[1]
model_training_dir = sys.argv[2]
summary_output_file = sys.argv[3]


log_files = sorted(glob.glob(model_training_dir +
	'full_rss_lf_model_train_test_tissue_' + test_tissue + '_*_training_log.txt'))

t = open(summary_output_file, 'w')
t.write('test_tissue\tlearning_rate\tl2_tissue_reg_strength\tl1_variant_reg_strength\tn_factors\t')
t.write('best_fit_iter\ttrain_loss\tval_loss\tval_corr\n')

n_written = 0
for log_file in log_files:
	params = parse_params_from_log_file(log_file, test_tissue)
	if params is None:
		print('Skipping (could not parse params): ' + log_file, flush=True)
		continue
	best_row = extract_best_epoch(log_file)
	if best_row is None:
		print('Skipping (no completed epochs): ' + log_file, flush=True)
		continue
	best_epoch, train_loss, val_loss, val_corr = best_row
	t.write(test_tissue + '\t' + params['lr'] + '\t' + params['l2t'] + '\t' +
		params['l1v'] + '\t' + params['n_factors'] + '\t')
	t.write(best_epoch + '\t' + str(train_loss) + '\t' + str(val_loss) + '\t' + str(val_corr) + '\n')
	n_written = n_written + 1

t.close()
print('Wrote summary for ' + str(n_written) + ' models to ' + summary_output_file, flush=True)
