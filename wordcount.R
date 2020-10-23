rmdFiles <- dir(pattern = 'Rmd$', ignore.case= TRUE)
rmdFiles <- rmdFiles[!grepl("000", rmdFiles)]
rmdFiles <- rmdFiles[!grepl('references', rmdFiles)]

counts <- sapply(rmdFiles, wordcountaddin::word_count, simplify = TRUE)

counts
sum(counts)
