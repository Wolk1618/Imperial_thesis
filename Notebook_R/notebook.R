library(R.utils)
library(gdata)
library(readxl)
library(scater)
library(scran)

count_matrix <- read_excel("Notebook R/GSE61533_HTSEQ_count_results.xls", sheet = 1)
sce <- SingleCellExperiment(assays = list(counts = count_matrix))
#print(dim(sce))


is_spike <- grepl("^ERCC", rownames(sce))
is_mito <- grepl("^mt-", rownames(sce))

sce <- perCellQCMetrics(sce, feature_controls = list(ERCC = is_spike, Mt = is_mito))
head(colnames(pData(sce)))
#
#isSpike(sce) <- "ERCC"
#
#par(mfrow = c(1, 2))
#hist(sce$total_counts / 1e6, xlab = "Library sizes (millions)", main = "",
#     breaks = 20, col = "grey80", ylab = "Number of cells")
#hist(sce$total_features, xlab = "Number of expressed genes", main = "",
#     breaks = 20, col = "grey80", ylab = "Number of cells")
#
#libsize.drop <- isOutlier(sce$total_counts, nmads = 3, type = "lower", log = TRUE)
#feature.drop <- isOutlier(sce$total_features, nmads = 3, type = "lower", log = TRUE)
#
#par(mfrow=c(1,2))
#hist(sce$pct_counts_feature_controls_Mt, xlab="Mitochondrial proportion (%)",
#     ylab="Number of cells", breaks=20, main="", col="grey80")
#hist(sce$pct_counts_feature_controls_ERCC, xlab="ERCC proportion (%)",
#     ylab="Number of cells", breaks=20, main="", col="grey80")
#
#mito.drop <- isOutlier(sce$pct_counts_feature_controls_Mt, nmads=3, type="higher")
#spike.drop <- isOutlier(sce$pct_counts_feature_controls_ERCC, nmads=3, type="higher")