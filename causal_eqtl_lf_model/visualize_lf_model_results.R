args = commandArgs(trailingOnly=TRUE)
library(cowplot)
library(ggplot2)
library(RColorBrewer)
options(warn=1)

figure_theme <- function() {
	return(theme(plot.title = element_text(face="plain",size=11), text = element_text(size=11),axis.text=element_text(size=11), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black"), legend.text = element_text(size=11), legend.title = element_text(size=11)))
}

mean_ci <- function(x) {
	x <- x[is.finite(x)]
	m <- mean(x)
	se <- sd(x) / sqrt(length(x))
	ci <- m + c(-1, 1) * qt(0.975, df = length(x) - 1) * se
	c(mean = m, lower_95 = ci[1], upper_95 = ci[2])
}

load_in_tissue_names <- function(borzoi_gtex_tissues_file) {
	tissue_df = read.table(borzoi_gtex_tissues_file, header=TRUE, sep="\t", stringsAsFactors=FALSE)
	return(as.character(tissue_df$tissue_name))
}

# Concatenate the per-test-tissue hyperparameter sweep summaries (output of summarize_lf_model_training_logs.py)
# Hyperparameter columns are kept as character so the exact strings (eg. "1e-3") can be used to rebuild training-log file names
load_hyperparameter_sweep_summary <- function(tissue_names, model_training_dir) {
	sweep_df = NULL
	for (tissue_name in tissue_names) {
		summary_file = paste0(model_training_dir, "training_summary_test_tissue_", tissue_name, ".txt")
		if (!file.exists(summary_file)) {
			print(paste0("Skipping (missing training summary): ", summary_file))
			next
		}
		tissue_df = read.table(summary_file, header=TRUE, sep="\t", stringsAsFactors=FALSE, na.strings=c("NA", "nan", "NaN"), colClasses=c(test_tissue="character", learning_rate="character", l2_tissue_reg_strength="character", l1_variant_reg_strength="character", n_factors="character"))
		if (nrow(tissue_df) == 0) {
			print(paste0("Skipping (no models in training summary): ", summary_file))
			next
		}
		sweep_df = rbind(sweep_df, tissue_df)
	}
	if (is.null(sweep_df)) {
		stop("No training summary files were loaded.")
	}
	# Order learning rates numerically
	learning_rates = unique(sweep_df$learning_rate)
	learning_rates = learning_rates[order(as.numeric(learning_rates))]
	sweep_df$learning_rate = factor(sweep_df$learning_rate, levels=learning_rates)
	# One line per (test tissue, non-learning-rate hyperparameter setting)
	sweep_df$config_group = paste(sweep_df$test_tissue, sweep_df$l2_tissue_reg_strength, sweep_df$l1_variant_reg_strength, sweep_df$n_factors, sep="_")
	return(sweep_df)
}

make_hyperparameter_sweep_plot <- function(sweep_df) {
	val_loss_plot <- ggplot(sweep_df, aes(x=learning_rate, y=val_loss)) +
	  geom_line(aes(group=config_group), color="grey60", alpha=.6, linewidth=.4) +
	  geom_point(color="#007C89", size=1.5) +
	  xlab("Learning rate") +
	  ylab("Best-epoch validation loss") +
	  figure_theme()

	val_corr_plot <- ggplot(sweep_df, aes(x=learning_rate, y=val_corr)) +
	  geom_line(aes(group=config_group), color="grey60", alpha=.6, linewidth=.4) +
	  geom_point(color="#007C89", size=1.5) +
	  xlab("Learning rate") +
	  ylab("Best-epoch validation correlation") +
	  figure_theme()

	joint <- plot_grid(val_loss_plot, val_corr_plot, labels=c("a", "b"), ncol=2)
	return(joint)
}

# Concatenate the best model's gene-level held-out test-tissue evaluations across test tissues
load_gene_level_evaluations <- function(tissue_names, model_training_dir) {
	gene_df = NULL
	for (tissue_name in tissue_names) {
		gene_eval_file = paste0(model_training_dir, "best_model_test_tissue_evaluation_", tissue_name, "_test_tissue_gene_evaluation.txt")
		if (!file.exists(gene_eval_file)) {
			print(paste0("Skipping (missing gene evaluation): ", gene_eval_file))
			next
		}
		tissue_gene_df = read.table(gene_eval_file, header=TRUE, sep="\t", stringsAsFactors=FALSE, na.strings=c("NA", "nan", "NaN"))
		# Genes where the test tissue is unobserved are reported as nan
		tissue_gene_df = tissue_gene_df[complete.cases(tissue_gene_df[, c("expr_corr", "marginal_std_effect_corr")]), ]
		if (nrow(tissue_gene_df) < 2) {
			print(paste0("Skipping (fewer than 2 evaluated genes): ", gene_eval_file))
			next
		}
		tissue_gene_df$tissue_name = tissue_name
		gene_df = rbind(gene_df, tissue_gene_df)
	}
	if (is.null(gene_df)) {
		stop("No gene-level evaluation files were loaded.")
	}
	return(gene_df)
}

# Per-tissue mean (+/- 95% CI) of the gene-level expression correlation and within-gene marginal effect correlation
summarize_per_tissue_gene_level_evaluations <- function(gene_df) {
	tissue_arr <- c()
	n_gene_arr <- c()
	expr_corr_mean_arr <- c()
	expr_corr_lb_arr <- c()
	expr_corr_ub_arr <- c()
	marg_corr_mean_arr <- c()
	marg_corr_lb_arr <- c()
	marg_corr_ub_arr <- c()
	for (tissue_name in unique(gene_df$tissue_name)) {
		tissue_gene_df = gene_df[gene_df$tissue_name == tissue_name, ]

		expr_corr_mean = mean(tissue_gene_df$expr_corr)
		expr_corr_se = sd(tissue_gene_df$expr_corr)/sqrt(length(tissue_gene_df$expr_corr))
		marg_corr_mean = mean(tissue_gene_df$marginal_std_effect_corr)
		marg_corr_se = sd(tissue_gene_df$marginal_std_effect_corr)/sqrt(length(tissue_gene_df$marginal_std_effect_corr))

		tissue_arr <- c(tissue_arr, tissue_name)
		n_gene_arr <- c(n_gene_arr, nrow(tissue_gene_df))
		expr_corr_mean_arr <- c(expr_corr_mean_arr, expr_corr_mean)
		expr_corr_lb_arr <- c(expr_corr_lb_arr, expr_corr_mean - (1.96*expr_corr_se))
		expr_corr_ub_arr <- c(expr_corr_ub_arr, expr_corr_mean + (1.96*expr_corr_se))
		marg_corr_mean_arr <- c(marg_corr_mean_arr, marg_corr_mean)
		marg_corr_lb_arr <- c(marg_corr_lb_arr, marg_corr_mean - (1.96*marg_corr_se))
		marg_corr_ub_arr <- c(marg_corr_ub_arr, marg_corr_mean + (1.96*marg_corr_se))
	}
	summary_df <- data.frame(
		tissue_name = tissue_arr,
		n_genes = n_gene_arr,
		expr_corr_mean = expr_corr_mean_arr,
		expr_corr_lb = expr_corr_lb_arr,
		expr_corr_ub = expr_corr_ub_arr,
		marginal_std_effect_corr_mean = marg_corr_mean_arr,
		marginal_std_effect_corr_lb = marg_corr_lb_arr,
		marginal_std_effect_corr_ub = marg_corr_ub_arr
	)
	return(summary_df)
}

# Per-tissue correlation between observed and predicted marginal effects, pooled across ALL variant-gene pairs
# (in contrast to the mean of within-gene correlations in summarize_per_tissue_gene_level_evaluations)
load_per_tissue_variant_level_correlation <- function(tissue_names, model_training_dir) {
	tissue_arr <- c()
	n_pair_arr <- c()
	corr_arr <- c()
	corr_lb_arr <- c()
	corr_ub_arr <- c()
	for (tissue_name in tissue_names) {
		variant_eval_file = paste0(model_training_dir, "best_model_test_tissue_evaluation_", tissue_name, "_test_tissue_variant_gene_pair_evaluation.txt")
		if (!file.exists(variant_eval_file)) {
			print(paste0("Skipping (missing variant evaluation): ", variant_eval_file))
			next
		}
		# These files are large: skip the gene_name / variant_name columns and only read the numeric ones
		vg_pair_df = read.table(variant_eval_file, header=TRUE, sep="\t", na.strings=c("NA", "nan", "NaN"), colClasses=c("NULL", "NULL", "numeric", "numeric", "numeric"), quote="", comment.char="")
		# Variant-gene pairs where the test tissue is unobserved are reported as nan
		vg_pair_df = vg_pair_df[complete.cases(vg_pair_df[, c("obs_marginal_std_effect", "pred_marginal_std_effect")]), ]
		n_pairs = nrow(vg_pair_df)
		if (n_pairs < 4) {
			print(paste0("Skipping (fewer than 4 evaluated variant-gene pairs): ", variant_eval_file))
			next
		}
		pooled_corr = cor(vg_pair_df$obs_marginal_std_effect, vg_pair_df$pred_marginal_std_effect)
		# Fisher-z 95% CI. Treats variant-gene pairs as independent (they are not, due to LD within a gene), so it is anti-conservative
		fisher_z = atanh(pooled_corr)
		fisher_z_se = 1.0/sqrt(n_pairs - 3)

		tissue_arr <- c(tissue_arr, tissue_name)
		n_pair_arr <- c(n_pair_arr, n_pairs)
		corr_arr <- c(corr_arr, pooled_corr)
		corr_lb_arr <- c(corr_lb_arr, tanh(fisher_z - (1.96*fisher_z_se)))
		corr_ub_arr <- c(corr_ub_arr, tanh(fisher_z + (1.96*fisher_z_se)))
	}
	if (length(tissue_arr) == 0) {
		stop("No variant-level evaluation files were loaded.")
	}
	pooled_corr_df <- data.frame(
		tissue_name = tissue_arr,
		n_variant_gene_pairs = n_pair_arr,
		pooled_corr = corr_arr,
		pooled_corr_lb = corr_lb_arr,
		pooled_corr_ub = corr_ub_arr
	)
	return(pooled_corr_df)
}

# One point (+/- CI) per test tissue; tissues displayed in the order given by tissue_order
make_per_tissue_correlation_pointrange_plot <- function(tissue_names, corr, corr_lb, corr_ub, tissue_order, y_label) {
	plot_df = data.frame(tissue_name=factor(tissue_names, levels=tissue_order), corr=corr, corr_lb=corr_lb, corr_ub=corr_ub)

	pp <- ggplot(plot_df, aes(x=tissue_name, y=corr)) +
	  geom_hline(yintercept=0, linetype="dashed", linewidth=.5, color="grey60") +
	  geom_pointrange(aes(ymin=corr_lb, ymax=corr_ub), color="#007C89", linewidth=.5, size=.3) +
	  coord_flip() +
	  figure_theme() +
	  theme(axis.text.y=element_text(size=7)) +
	  labs(x="Test tissue", y=y_label)
	return(pp)
}

# Distribution (violin + box) of a gene-level correlation across genes, per test tissue
make_per_tissue_correlation_distribution_plot <- function(gene_df, corr_col, tissue_order, y_label) {
	plot_df = data.frame(tissue_name=factor(gene_df$tissue_name, levels=tissue_order), corr=gene_df[[corr_col]])

	pp <- ggplot(plot_df, aes(x=tissue_name, y=corr)) +
	  geom_hline(yintercept=0, linetype="dashed", linewidth=.5, color="grey60") +
	  geom_violin(fill="#007C89", color=NA, alpha=.45, scale="width") +
	  geom_boxplot(width=.25, fill="white", color="black", linewidth=.3, outlier.size=.3, outlier.alpha=.3) +
	  coord_flip() +
	  figure_theme() +
	  theme(axis.text.y=element_text(size=7)) +
	  labs(x="Test tissue", y=y_label)
	return(pp)
}

make_per_tissue_expr_corr_vs_marginal_effect_corr_scatter <- function(summary_df) {
	pp <- ggplot(summary_df, aes(x=marginal_std_effect_corr_mean, y=expr_corr_mean)) +
	  geom_segment(aes(x=marginal_std_effect_corr_lb, xend=marginal_std_effect_corr_ub, y=expr_corr_mean, yend=expr_corr_mean), linewidth=.3, alpha=.8) +
	  geom_segment(aes(x=marginal_std_effect_corr_mean, xend=marginal_std_effect_corr_mean, y=expr_corr_lb, yend=expr_corr_ub), linewidth=.3, alpha=.8) +
	  geom_point(size=2, color="#007C89") +
	  xlab("Mean marginal effect correlation") +
	  ylab("Mean expression correlation") +
	  figure_theme()
	return(pp)
}

# Histogram of a gene-level correlation across genes (single tissue); dashed red line at the mean
make_correlation_histogram <- function(corr_values, x_label) {
	plot_df = data.frame(corr=corr_values)
	pp <- ggplot(plot_df, aes(x=corr)) +
	  geom_histogram(bins=40, fill="#007C89", color="white", linewidth=.2) +
	  geom_vline(xintercept=0, linetype="dashed", linewidth=.5, color="grey60") +
	  geom_vline(xintercept=mean(corr_values), linetype="dashed", linewidth=.5, color="firebrick") +
	  figure_theme() +
	  labs(x=x_label, y="Number of genes")
	return(pp)
}

make_variant_level_marginal_effect_scatter <- function(variant_eval_file, tissue_name) {
	vg_pair_df = read.table(variant_eval_file, header=TRUE, sep="\t", stringsAsFactors=FALSE, na.strings=c("NA", "nan", "NaN"))
	vg_pair_df = vg_pair_df[complete.cases(vg_pair_df[, c("obs_marginal_std_effect", "pred_marginal_std_effect")]), ]

	pp <- ggplot(vg_pair_df, aes(x=pred_marginal_std_effect, y=obs_marginal_std_effect)) +
	  geom_point(size=.25, alpha=.06) +
	  geom_abline(intercept=0, slope=1, linetype="dashed", color="firebrick") +
	  geom_smooth(method="lm", se=FALSE, color="dodgerblue4", linewidth=.6) +
	  xlab("Predicted marginal standardized effect") +
	  ylab("Observed marginal standardized effect") +
	  ggtitle(tissue_name) +
	  figure_theme()
	return(pp)
}

# Per-epoch train loss / validation loss / validation correlation for every sweep config of a single test tissue
make_training_curves_plot <- function(tissue_sweep_df, tissue_name, model_training_dir) {
	if (nrow(tissue_sweep_df) == 0) {
		return(NULL)
	}
	curves_df = NULL
	for (row_iter in 1:nrow(tissue_sweep_df)) {
		log_file = paste0(model_training_dir, "full_rss_lf_model_train_test_tissue_", tissue_name, "_lr_", as.character(tissue_sweep_df$learning_rate[row_iter]), "_l2t_", tissue_sweep_df$l2_tissue_reg_strength[row_iter], "_l1v_", tissue_sweep_df$l1_variant_reg_strength[row_iter], "_var_arch_K_", tissue_sweep_df$n_factors[row_iter], "_training_log.txt")
		if (!file.exists(log_file)) {
			print(paste0("Skipping (missing training log): ", log_file))
			next
		}
		log_df = read.table(log_file, header=TRUE, sep="\t", stringsAsFactors=FALSE, na.strings=c("NA", "nan", "NaN"))
		log_df$learning_rate = as.character(tissue_sweep_df$learning_rate[row_iter])
		log_df$config_group = tissue_sweep_df$config_group[row_iter]
		# One line per training run: config_group deliberately excludes the learning rate (it is the x-axis of the sweep plot), so add it back here
		log_df$run_group = paste(log_df$learning_rate, log_df$config_group, sep="_")
		curves_df = rbind(curves_df, log_df[, c("epoch", "train_loss", "val_loss", "val_corr", "learning_rate", "config_group", "run_group")])
	}
	if (is.null(curves_df)) {
		return(NULL)
	}
	curves_df$learning_rate = factor(curves_df$learning_rate, levels=levels(tissue_sweep_df$learning_rate))

	train_loss_plot <- ggplot(curves_df, aes(x=epoch, y=train_loss, color=learning_rate, group=run_group)) +
	  geom_line(linewidth=.5) +
	  scale_color_brewer(palette="Dark2", name="Learning rate") +
	  xlab("Epoch") +
	  ylab("Training loss") +
	  figure_theme()

	val_loss_plot <- ggplot(curves_df, aes(x=epoch, y=val_loss, color=learning_rate, group=run_group)) +
	  geom_line(linewidth=.5) +
	  scale_color_brewer(palette="Dark2", name="Learning rate") +
	  xlab("Epoch") +
	  ylab("Validation loss") +
	  figure_theme()

	val_corr_plot <- ggplot(curves_df, aes(x=epoch, y=val_corr, color=learning_rate, group=run_group)) +
	  geom_line(linewidth=.5) +
	  scale_color_brewer(palette="Dark2", name="Learning rate") +
	  xlab("Epoch") +
	  ylab("Validation correlation") +
	  figure_theme()

	legend_source_plot <- train_loss_plot + theme(legend.position="bottom")
	# ggplot2 >= 3.5 names the bottom guide box "guide-box-bottom"; older versions have a single "guide-box"
	legend <- tryCatch(get_plot_component(legend_source_plot, "guide-box-bottom"), error=function(e) NULL)
	if (is.null(legend) || inherits(legend, "zeroGrob")) {
		legend <- get_plot_component(legend_source_plot, "guide-box")
	}
	panels <- plot_grid(train_loss_plot + theme(legend.position="none"), val_loss_plot + theme(legend.position="none"), val_corr_plot + theme(legend.position="none"), labels=c("a", "b", "c"), ncol=3)
	joint <- plot_grid(ggdraw() + draw_label(tissue_name, size=11), panels, legend, ncol=1, rel_heights=c(.08, 1, .12))
	return(joint)
}





###################
# Command line args
#####################
borzoi_gtex_tissues_file = args[1]
model_training_dir = args[2]
visualization_dir = args[3]

# Tissues used for the single-tissue (gene-level correlation histograms, variant-level scatter, training curve) plots
example_tissues = c("Adipose_Subcutaneous", "Whole_Blood", "Muscle_Skeletal", "Liver")


tissue_names = load_in_tissue_names(borzoi_gtex_tissues_file)


########################
# Hyperparameter sweep: best-epoch validation loss / correlation of every config, for every test tissue
########################
sweep_df = load_hyperparameter_sweep_summary(tissue_names, model_training_dir)
print(sweep_df)

sweep_plot <- make_hyperparameter_sweep_plot(sweep_df)
output_file <- paste0(visualization_dir, "hyperparameter_sweep_validation_by_learning_rate.pdf")
ggsave(sweep_plot, file=output_file, width=7.2, height=3.2, units="in")


########################
# Held-out test-tissue performance of the best (lowest val loss) model, per test tissue (gene-level metrics)
########################
gene_df = load_gene_level_evaluations(tissue_names, model_training_dir)
per_tissue_summary_df = summarize_per_tissue_gene_level_evaluations(gene_df)
print(per_tissue_summary_df)
print("Cross-tissue mean of per-tissue mean expression correlation:")
print(mean_ci(per_tissue_summary_df$expr_corr_mean))
print("Cross-tissue mean of per-tissue mean within-gene marginal effect correlation:")
print(mean_ci(per_tissue_summary_df$marginal_std_effect_corr_mean))

per_tissue_plot_height = max(3.0, 0.13*nrow(per_tissue_summary_df) + 1.0)

# Tissue orderings (by per-tissue mean) shared between the mean plots and the distribution plots
expr_corr_tissue_order = per_tissue_summary_df$tissue_name[order(per_tissue_summary_df$expr_corr_mean)]
marg_corr_tissue_order = per_tissue_summary_df$tissue_name[order(per_tissue_summary_df$marginal_std_effect_corr_mean)]

# Expression correlation: per-tissue mean (+/- 95% CI) and per-tissue distribution across genes
per_tissue_expr_corr_plot <- make_per_tissue_correlation_pointrange_plot(per_tissue_summary_df$tissue_name, per_tissue_summary_df$expr_corr_mean, per_tissue_summary_df$expr_corr_lb, per_tissue_summary_df$expr_corr_ub, expr_corr_tissue_order, "Mean expression correlation\n(held-out test tissue)")
output_file <- paste0(visualization_dir, "per_tissue_test_tissue_mean_expr_corr.pdf")
ggsave(per_tissue_expr_corr_plot, file=output_file, width=4.5, height=per_tissue_plot_height, units="in")

per_tissue_expr_corr_distribution_plot <- make_per_tissue_correlation_distribution_plot(gene_df, "expr_corr", expr_corr_tissue_order, "Expression correlation across genes\n(held-out test tissue)")
output_file <- paste0(visualization_dir, "per_tissue_test_tissue_expr_corr_distribution.pdf")
ggsave(per_tissue_expr_corr_distribution_plot, file=output_file, width=5.0, height=per_tissue_plot_height, units="in")

# Within-gene marginal effect correlation: per-tissue mean (+/- 95% CI) and per-tissue distribution across genes
per_tissue_marg_corr_plot <- make_per_tissue_correlation_pointrange_plot(per_tissue_summary_df$tissue_name, per_tissue_summary_df$marginal_std_effect_corr_mean, per_tissue_summary_df$marginal_std_effect_corr_lb, per_tissue_summary_df$marginal_std_effect_corr_ub, marg_corr_tissue_order, "Mean within-gene marginal effect correlation\n(held-out test tissue)")
output_file <- paste0(visualization_dir, "per_tissue_test_tissue_mean_marginal_std_effect_corr.pdf")
ggsave(per_tissue_marg_corr_plot, file=output_file, width=4.5, height=per_tissue_plot_height, units="in")

per_tissue_marg_corr_distribution_plot <- make_per_tissue_correlation_distribution_plot(gene_df, "marginal_std_effect_corr", marg_corr_tissue_order, "Within-gene marginal effect correlation across genes\n(held-out test tissue)")
output_file <- paste0(visualization_dir, "per_tissue_test_tissue_marginal_std_effect_corr_distribution.pdf")
ggsave(per_tissue_marg_corr_distribution_plot, file=output_file, width=5.0, height=per_tissue_plot_height, units="in")

per_tissue_scatter <- make_per_tissue_expr_corr_vs_marginal_effect_corr_scatter(per_tissue_summary_df)
output_file <- paste0(visualization_dir, "per_tissue_test_tissue_expr_corr_vs_marginal_std_effect_corr_scatter.pdf")
ggsave(per_tissue_scatter, file=output_file, width=4.2, height=4, units="in")


########################
# Marginal effect correlation pooled across ALL variant-gene pairs, per test tissue
########################
per_tissue_pooled_corr_df = load_per_tissue_variant_level_correlation(tissue_names, model_training_dir)
print(per_tissue_pooled_corr_df)
print("Cross-tissue mean of per-tissue pooled (all variant-gene pair) marginal effect correlation:")
print(mean_ci(per_tissue_pooled_corr_df$pooled_corr))

pooled_corr_tissue_order = per_tissue_pooled_corr_df$tissue_name[order(per_tissue_pooled_corr_df$pooled_corr)]
per_tissue_pooled_corr_plot <- make_per_tissue_correlation_pointrange_plot(per_tissue_pooled_corr_df$tissue_name, per_tissue_pooled_corr_df$pooled_corr, per_tissue_pooled_corr_df$pooled_corr_lb, per_tissue_pooled_corr_df$pooled_corr_ub, pooled_corr_tissue_order, "Marginal effect correlation across all\nvariant-gene pairs (held-out test tissue)")
output_file <- paste0(visualization_dir, "per_tissue_test_tissue_all_variant_gene_pair_marginal_std_effect_corr.pdf")
ggsave(per_tissue_pooled_corr_plot, file=output_file, width=4.5, height=max(3.0, 0.13*nrow(per_tissue_pooled_corr_df) + 1.0), units="in")


########################
# Single-tissue plots (gene-level correlation histograms, variant-level observed vs predicted marginal effects, training curves) for each example tissue
########################
for (example_tissue in example_tissues[!(example_tissues %in% per_tissue_summary_df$tissue_name)]) {
	print(paste0("Skipping example tissue (not evaluated): ", example_tissue))
}
evaluated_example_tissues = example_tissues[example_tissues %in% per_tissue_summary_df$tissue_name]
if (length(evaluated_example_tissues) == 0) {
	evaluated_example_tissues = per_tissue_summary_df$tissue_name[1]
	print(paste0("None of the example tissues were evaluated; using ", evaluated_example_tissues, " instead"))
}

for (example_tissue in evaluated_example_tissues) {
	# Histograms of gene-level expression correlation and within-gene marginal effect correlation
	example_gene_df = gene_df[gene_df$tissue_name == example_tissue, ]
	expr_corr_histogram <- make_correlation_histogram(example_gene_df$expr_corr, "Expression correlation")
	marg_corr_histogram <- make_correlation_histogram(example_gene_df$marginal_std_effect_corr, "Within-gene marginal effect correlation")
	histogram_panels <- plot_grid(expr_corr_histogram, marg_corr_histogram, labels=c("a", "b"), ncol=2)
	histogram_plot <- plot_grid(ggdraw() + draw_label(example_tissue, size=11), histogram_panels, ncol=1, rel_heights=c(.08, 1))
	output_file <- paste0(visualization_dir, example_tissue, "_gene_level_correlation_histograms.pdf")
	ggsave(histogram_plot, file=output_file, width=7.2, height=3.0, units="in")

	# Variant-level observed vs predicted marginal effects
	variant_eval_file = paste0(model_training_dir, "best_model_test_tissue_evaluation_", example_tissue, "_test_tissue_variant_gene_pair_evaluation.txt")
	if (file.exists(variant_eval_file)) {
		variant_scatter <- make_variant_level_marginal_effect_scatter(variant_eval_file, example_tissue)
		output_file <- paste0(visualization_dir, example_tissue, "_variant_level_marginal_std_effect_scatter.pdf")
		ggsave(variant_scatter, file=output_file, width=4.2, height=4, units="in")
	} else {
		print(paste0("Skipping (missing variant evaluation): ", variant_eval_file))
	}

	# Training curves across the hyperparameter sweep
	training_curves_plot <- make_training_curves_plot(sweep_df[sweep_df$test_tissue == example_tissue, ], example_tissue, model_training_dir)
	if (!is.null(training_curves_plot)) {
		output_file <- paste0(visualization_dir, example_tissue, "_training_curves.pdf")
		ggsave(training_curves_plot, file=output_file, width=7.2, height=2.9, units="in")
	}
}
