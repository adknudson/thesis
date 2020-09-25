library(rethinking)

m1 <- readRDS("scratch/av_m1.Rds")
m2 <- readRDS("scratch/av_m2.Rds")
m3 <- readRDS("scratch/av_m3.Rds")
m4 <- readRDS("scratch/av_m4.Rds")
m6 <- readRDS("scratch/av_m6.Rds")

compare(m1, m2)

compare(m1, m2, m3, m4, m6)

compare(av_m2, av_m4)

compare(av_m3, av_m6)

