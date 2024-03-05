install.packages("gdata")
install.packages("readxl")

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
#BiocManager::install("scater")
#BiocManager::install("simpleSingleCell")
BiocManager::install("scran")

