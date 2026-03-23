args = commandArgs(trailingOnly=TRUE)
library(cowplot)
library(ggplot2)
library(hash)
library(RColorBrewer)
options(warn=1)

figure_theme <- function() {
	return(theme(plot.title = element_text(face="plain",size=11), text = element_text(size=11),axis.text=element_text(size=11), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black"), legend.text = element_text(size=11), legend.title = element_text(size=11)))
}




make_factorization_vs_nn_loss_scatter <- function(df) {
pp <- ggplot(data = df, aes(x = nn_loss, y = loss)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  figure_theme() +
  labs(x = "Loss: closest tissue", y = "Loss: factorization")

return(pp)
}

make_factorization_vs_nn_correlation_scatter <- function(df) {
pp <- ggplot(data = df, aes(x = nn_correlation, y = correlation)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  figure_theme() +
  labs(x = "correlation: closest tissue", y = "correlation: factorization")

return(pp)
}

make_sample_size_vs_loss_scatter <- function(df) {
pp <- ggplot(data = df, aes(x = sample_size, y = loss)) +
  geom_point() +
  figure_theme() +
  labs(x = "Sample size", y = "Loss: factorization")
}


make_pred_var_vs_resid_var_scatter <- function(df) {
  pp <- ggplot(data = df, aes(x = pred_resid_var, y = resid_var)) +
    geom_point() +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    figure_theme() +
    labs(x = "QTL-pred estimated residual variance", y = "Average residual variance")
  return(pp)
}


make_average_loss_barplot <- function(df, df2) {

  # QTL Pred
  qtl_pred_mean_loss <- mean(df$loss)
  qtl_pred_mean_loss_se <- sd(df$loss)/sqrt(length(df$loss))
  qtl_pred_mean_loss_lb <- qtl_pred_mean_loss - (1.96*qtl_pred_mean_loss_se)
  qtl_pred_mean_loss_ub <- qtl_pred_mean_loss + (1.96*qtl_pred_mean_loss_se)

  # Nearest tissue
  nt_mean_loss <- mean(df$nn_loss)
  nt_mean_loss_se <- sd(df$nn_loss)/sqrt(length(df$nn_loss))
  nt_mean_loss_lb <- nt_mean_loss - (1.96*nt_mean_loss_se)
  nt_mean_loss_ub <- nt_mean_loss + (1.96*nt_mean_loss_se)
 
  # Random tissue
  rt_mean_loss <- mean(df2$rt_loss)
  rt_mean_loss_se <- sd(df2$rt_loss)/sqrt(length(df2$rt_loss))
  rt_mean_loss_lb <- rt_mean_loss - (1.96*rt_mean_loss_se)
  rt_mean_loss_ub <- rt_mean_loss + (1.96*rt_mean_loss_se)


  methods <- c("QTL-pred        ", "Closest tissue  ", "Random tissue")
  mean_loss <- c(qtl_pred_mean_loss, nt_mean_loss, rt_mean_loss)
  mean_loss_lb <- c(qtl_pred_mean_loss_lb, nt_mean_loss_lb, rt_mean_loss_lb)
  mean_loss_ub <- c(qtl_pred_mean_loss_ub, nt_mean_loss_ub, rt_mean_loss_ub)

  df <- data.frame(
    method = methods,
    mean_loss = mean_loss,
    mean_loss_lb = mean_loss_lb,
    mean_loss_ub = mean_loss_ub
  )

  print(df)

df$method <- factor(df$method, levels = df$method)

pp <- ggplot(df, aes(x = method, y = mean_loss, fill = method)) +
  geom_col(width = 0.75) +
  geom_errorbar(
    aes(ymin = mean_loss_lb, ymax = mean_loss_ub),
    width = 0.2,
    linewidth = 0.7
  ) +
geom_text(
  aes(label = method),
  y = 1.53,
  angle = 90,
  vjust = 0,
  hjust = 0.5,
  position = position_nudge(x = .075),  # force exact bar center
  color = "black",
  size = 4
)+
  scale_fill_manual(values = c(
    "QTL-pred        " = "#1b9e77",
    "Closest tissue  " = "#d95f02",
    "Random tissue" = "#7570b3"
  )) +
  labs(x = NULL, y = "Avgerage test loss") +
  figure_theme() +
  theme(
    legend.position = "none",
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  )
  return(pp)

}

make_scatter <- function(per_tissue_df, titler) {

  pp <- ggplot(data = per_tissue_df, aes(x = pred_beta, y = beta)) +
    geom_point(alpha=.1,size=.05, color="#1b9e77") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    figure_theme() +
    labs(x = "QTL-pred effect estimate", y = "eQTL effect estimate", title=titler)
  return(pp)
}

make_hex_scatter <- function(per_tissue_df, titler, bins = 60) {
  ggplot(per_tissue_df, aes(x = pred_beta, y = beta)) +
    geom_hex(bins = bins) +
    scale_fill_viridis_c(
      trans = "log10",
      name = "Count",
      breaks = scales::log_breaks()
    ) +
    geom_abline(slope = 1, intercept = 0,
                linetype = "dashed", color = "red") +
    figure_theme() +
    labs(
      x = "QTL-pred effect size",
      y = "eQTL effect size",
      title = titler
    ) +
    theme(
      plot.title = element_text(hjust = 0.5)
    )
}

make_heat_scatter <- function(per_tissue_df, titler, bins = 80) {
  ggplot(per_tissue_df, aes(x = pred_beta, y = beta)) +
    geom_bin2d(bins = bins) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    figure_theme() +
    labs(
      x = "QTL-pred effect estimate",
      y = "eQTL effect estimate",
      title = titler,
      fill = "Count"
    )
}

################
# Command line args
organized_results_file = args[1]
organized_results_file2 = args[2]

visualization_dir = args[3]



# Load in data
df = read.table(organized_results_file, header=TRUE, sep="\t")

df2 = read.table(organized_results_file2, header=TRUE, sep="\t")

if (FALSE) {
indices = (as.character(df$tissue) != "Thyroid") & (df$pred_resid_var < 0.01)
df <- df[indices,]
}


# Make average loss bar plot
avg_loss_barplot <- make_average_loss_barplot(df, df2)
output_file <- paste0(visualization_dir, "avg_loss_bar_plot.pdf")
ggsave(avg_loss_barplot, file=output_file, width=7.2/3, height=2.9, units="in")



tissue_name="Lung"
per_tissue_results_file <- paste0("/lab-share/CHIP-Strober-e2/Public/ben/unseen_qtl_prediction/marginal_predictions/modeling_training/expression_reduced_eqtls_nPCs_20_seed1_test_tissue_", tissue_name,"_het_var_multi_restart_test_preds.txt")
per_tissue_df <- read.table(per_tissue_results_file, header=TRUE)
scatter <- make_hex_scatter(per_tissue_df, paste0(tissue_name," tissue"))


joint <- plot_grid(scatter + theme(legend.position="none"), avg_loss_barplot, ncol=2, rel_widths=c(1,.85), labels=c("a", "b"))
output_file <- paste0(visualization_dir, "r01_plot.pdf")
ggsave(joint, file=output_file, width=2*7.2/3, height=2.95, units="in")



# Make pred var vs resid var scatter
if (FALSE) {
pp <- make_pred_var_vs_resid_var_scatter(df)
output_file <- paste0(visualization_dir, "pred_var_vs_resid_var_scatter.pdf")
ggsave(pp, file=output_file, width=7.2, height=4.5, units="in")
}


if (FALSE) {

unique_tissues <- unique(as.character(df$tissue))

for (tiss_iter in 1:length(unique_tissues)) {

tissue_name <- unique_tissues[tiss_iter]

if (tissue_name != "Bladder") {

per_tissue_results_file <- paste0("/lab-share/CHIP-Strober-e2/Public/ben/unseen_qtl_prediction/marginal_predictions/modeling_training/expression_reduced_eqtls_nPCs_20_seed1_test_tissue_", tissue_name,"_het_var_multi_restart_test_preds.txt")
per_tissue_df <- read.table(per_tissue_results_file, header=TRUE)
pp <- make_hex_scatter(per_tissue_df, paste0(tissue_name," eQTLs"))
output_file <- paste0(visualization_dir, tissue_name,"_scatter.pdf")
ggsave(pp + theme(legend.position="none"), file=output_file, width=7.2/3, height=3.2, units="in")
}
}
}




if (FALSE) {
# Make loss scatter
pp <- make_factorization_vs_nn_loss_scatter(df)
output_file <- paste0(visualization_dir, "factorization_vs_nn_loss_scatter.pdf")
ggsave(pp, file=output_file, width=7.2, height=4.5, units="in")

# Make correlation scatter
pp <- make_factorization_vs_nn_correlation_scatter(df)
output_file <- paste0(visualization_dir, "factorization_vs_nn_correlation_scatter.pdf")
ggsave(pp, file=output_file, width=7.2, height=4.5, units="in")

# Make sample size vs loss scatter
pp <- make_sample_size_vs_loss_scatter(df)
output_file <- paste0(visualization_dir, "factorization_loss_vs_sample_size.pdf")
ggsave(pp, file=output_file, width=7.2, height=4.5, units="in")

}