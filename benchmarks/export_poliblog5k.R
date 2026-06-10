#!/usr/bin/env Rscript
# Export stm's bundled poliblog5k corpus to a flat CSV so bench.py can load it
# without an R dependency at every run.
#
# Output path is passed as the first command-line argument; when not given,
# defaults to poliblog5k_prepped.csv in the script's own directory.
# Columns: rating, day, blog, text  (text is the reconstructed bag-of-words string)
#
# Run once (bench.py calls it automatically when the CSV is missing):
#   Rscript benchmarks/export_poliblog5k.R /path/to/poliblog5k_prepped.csv

suppressMessages(library(stm))
data(poliblog5k, package = "stm")

voc <- poliblog5k.voc
# Reconstruct each document as a space-joined token string from the stm sparse
# format: d[1,] = vocabulary indices (1-based), d[2,] = token counts.
txt <- vapply(poliblog5k.docs, function(d) {
  paste(rep(voc[d[1, ]], d[2, ]), collapse = " ")
}, character(1))

out <- data.frame(
  rating = poliblog5k.meta$rating,
  day    = poliblog5k.meta$day,
  blog   = poliblog5k.meta$blog,
  text   = txt,
  stringsAsFactors = FALSE
)

# Drop empty documents (edge case in corpus).
out <- out[nchar(out$text) > 0, ]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) {
  dest <- args[1]
} else {
  # Fallback: write to the same directory as this script if invoked directly.
  # When called via bench.py the output path is always passed explicitly.
  dest <- "poliblog5k_prepped.csv"
}

write.csv(out, dest, row.names = FALSE)
cat("wrote", nrow(out), "docs to", dest, "\n")
