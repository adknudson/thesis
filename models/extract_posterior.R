m044s_av  <- readRDS("models/m044s_av.rds")
m044s_vis <- readRDS("models/m044s_vis.rds")
m044s_dur <- readRDS("models/m044s_dur.rds")
m044s_sm  <- readRDS("models/m044s_sm.rds")

p044s_av  <- rstan::extract(m044s_av)
p044s_vis <- rstan::extract(m044s_vis)
p044s_dur <- rstan::extract(m044s_dur)
p044s_sm  <- rstan::extract(m044s_sm)

saveRDS(p044s_av,  "models/p044s_av.rds")
saveRDS(p044s_vis, "models/p044s_vis.rds")
saveRDS(p044s_dur, "models/p044s_dur.rds")
saveRDS(p044s_sm,  "models/p044s_sm.rds")
