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


################
# Command line args
organized_results_file = args[1]
visualization_dir = args[2]


print(organized_results_file)

# Load in data
df = read.table(organized_results_file, header=TRUE, sep="\t")

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