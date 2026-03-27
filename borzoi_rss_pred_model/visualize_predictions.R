args = commandArgs(trailingOnly=TRUE)
library(cowplot)
library(ggplot2)
library(RColorBrewer)
options(warn=1)

figure_theme <- function() {
	return(theme(plot.title = element_text(face="plain",size=11), text = element_text(size=11),axis.text=element_text(size=11), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black"), legend.text = element_text(size=11), legend.title = element_text(size=11)))
}
r35_figure_theme <- function() {
	return(theme(plot.title = element_text(face="plain",size=11), text = element_text(size=9),axis.text=element_text(size=9), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black"), legend.text = element_text(size=9), legend.title = element_text(size=9)))
}

make_borzoi_vs_factor_model_scatter <- function(expr_pred_summary_file, visualization_dir) {
	expr_pred_summary_df = read.table(expr_pred_summary_file, header=TRUE, sep="\t", stringsAsFactors=FALSE)

	borzoi_df = expr_pred_summary_df[expr_pred_summary_df$method == "borzoi", c("tissue_name", "mean", "mean_se")]
	factor_model_df = expr_pred_summary_df[expr_pred_summary_df$method == "factor_model", c("tissue_name", "mean", "mean_se")]

	colnames(borzoi_df) = c("tissue_name", "borzoi_mean", "borzoi_mean_se")
	colnames(factor_model_df) = c("tissue_name", "factor_model_mean", "factor_model_mean_se")

	plot_df = merge(borzoi_df, factor_model_df, by="tissue_name")

	plot_df$borzoi_ci_lb = plot_df$borzoi_mean - 1.96*plot_df$borzoi_mean_se
	plot_df$borzoi_ci_ub = plot_df$borzoi_mean + 1.96*plot_df$borzoi_mean_se
	plot_df$factor_model_ci_lb = plot_df$factor_model_mean - 1.96*plot_df$factor_model_mean_se
	plot_df$factor_model_ci_ub = plot_df$factor_model_mean + 1.96*plot_df$factor_model_mean_se

	axis_min = min(c(plot_df$borzoi_ci_lb, plot_df$factor_model_ci_lb))
	axis_max = max(c(plot_df$borzoi_ci_ub, plot_df$factor_model_ci_ub))

	pp <- ggplot(plot_df, aes(x=borzoi_mean, y=factor_model_mean)) +
	  geom_segment(aes(x=borzoi_ci_lb, xend=borzoi_ci_ub, y=factor_model_mean, yend=factor_model_mean), linewidth=.3, alpha=.8) +
	  geom_segment(aes(x=borzoi_mean, xend=borzoi_mean, y=factor_model_ci_lb, yend=factor_model_ci_ub), linewidth=.3, alpha=.8) +
	  geom_point(size=2) +
	  geom_abline(intercept=0, slope=1, linetype="dashed", color="firebrick") +
	  xlab("Borzoi mean predicted\nexpression correlation") +
	  ylab("Factor model mean predicted\nexpression correlation") +
	  coord_equal(xlim=c(axis_min, axis_max), ylim=c(axis_min, axis_max)) +
	  figure_theme()

	output_file <- paste0(visualization_dir, "borzoi_vs_factor_model_expr_corr_scatter.pdf")
	ggsave(output_file, pp, width=4.2, height=4, units="in")
}

make_variant_level_zed_scatter <- function(single_tissue_vg_pair_file, tissue_name, visualization_dir) {
	vg_pair_df = read.table(single_tissue_vg_pair_file, header=TRUE, sep="\t", stringsAsFactors=FALSE)
	vg_pair_df = vg_pair_df[vg_pair_df$gene_split == "test", ]
	vg_pair_df$obs_gene_zed = as.numeric(vg_pair_df$obs_gene_zed)
	vg_pair_df$pred_gene_zed = as.numeric(vg_pair_df$pred_gene_zed)
	vg_pair_df = vg_pair_df[complete.cases(vg_pair_df[, c("obs_gene_zed", "pred_gene_zed")]), ]

	pp <- ggplot(vg_pair_df, aes(x=pred_gene_zed, y=obs_gene_zed)) +
	  geom_point(size=.25, alpha=.06) +
	  geom_abline(intercept=0, slope=1, linetype="dashed", color="firebrick") +
	  geom_smooth(method="lm", se=FALSE, color="dodgerblue4", linewidth=.6) +
	  xlab("Predicted gene z-score") +
	  ylab("Observed gene z-score") +
	  ggtitle(tissue_name) +
	  figure_theme()

	output_file <- paste0(visualization_dir, tissue_name, "_variant_level_zed_scatter.pdf")
	ggsave(output_file, pp, width=4.2, height=4, units="in")
}











###################
# Command line args
#####################
borzoi_gtex_tissues_file = args[1]
model_training_dir = args[2]
visualization_dir = args[3]

expr_pred_summary_file = paste0(model_training_dir, "results_summary.txt")
make_borzoi_vs_factor_model_scatter(expr_pred_summary_file, visualization_dir)

tissue_name="Adipose_Subcutaneous"
single_tissue_vg_pair_file = paste0(model_training_dir,"full_rss_model_held_out_genes_eval_train_test_tissue_", tissue_name, "_lr_1e-5_l2t_100.0_l2v_100.0_var_arch_128x64x32_all_variant_gene_pairs_test_tissue_evaluation.txt")
make_variant_level_zed_scatter(single_tissue_vg_pair_file, tissue_name, visualization_dir)
