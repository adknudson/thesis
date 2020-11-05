m034s_av  <- readRDS("models/m034s_av.rds")
m034s_vis <- readRDS("models/m034s_vis.rds")
m034s_dur <- readRDS("models/m034s_dur.rds")
m034s_sm  <- readRDS("models/m034s_sm.rds")

p034s_av  <- rstan::extract(m034s_av)
p034s_vis <- rstan::extract(m034s_vis)
p034s_dur <- rstan::extract(m034s_dur)
p034s_sm  <- rstan::extract(m034s_sm)

saveRDS(p034s_av,  "models/p034s_av.rds")
saveRDS(p034s_vis, "models/p034s_vis.rds")
saveRDS(p034s_dur, "models/p034s_dur.rds")
saveRDS(p034s_sm,  "models/p034s_sm.rds")
