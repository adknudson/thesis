library(loo)
options(mc.cores = 12)

m044s_av  <- readRDS("~/Projects/thesis/models/m044s_av.rds")
m046_av   <- readRDS("~/Projects/thesis/models/m046_av.rds")
m044s_vis <- readRDS("~/Projects/thesis/models/m044s_vis.rds")
m046_vis  <- readRDS("~/Projects/thesis/models/m046_vis.rds")

m044s_dur <- readRDS("~/Projects/thesis/models/m044s_dur.rds")
m046_dur  <- readRDS("~/Projects/thesis/models/m046_dur.rds")
m044s_sm  <- readRDS("~/Projects/thesis/models/m044s_sm.rds")
m046_sm   <- readRDS("~/Projects/thesis/models/m046_sm.rds")

l044s_vis <- loo(m044s_vis)
l044s_av  <- loo(m044s_av )
l046_vis  <- loo(m046_vis )
l046_av   <- loo(m046_av  )

l044s_dur <- loo(m044s_dur)
l046_dur  <- loo(m046_dur )
l044s_sm  <- loo(m044s_sm )
l046_sm   <- loo(m046_sm  )

comp_av  <- loo_compare(l044s_av,  l046_av)
comp_vis <- loo_compare(l044s_vis, l046_vis)
comp_dur <- loo_compare(l044s_dur, l046_dur)
comp_sm  <- loo_compare(l044s_sm,  l046_sm)

print(comp_av,  simplify = FALSE)
print(comp_vis, simplify = FALSE)
print(comp_dur, simplify = FALSE)
print(comp_sm,  simplify = FALSE)
