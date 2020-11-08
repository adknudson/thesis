fp <- list.files(path = "data/RecalibrationData/",
                 recursive = TRUE,
                 pattern = "(.mat)$")
fp_typ_reg <-
  "^(\\w+)/(\\w+)/(\\w+)/[A-Z]{2,3}_*[A-Z]*(adapt[0-9]|baseline[0-9]*).*"
fp_typ_logi <- str_detect(fp, fp_typ_reg)
fp_typ <- fp[fp_typ_logi]
fp_atyp <- fp[!fp_typ_logi]
get_feat_typ <- function(mat) {
  expr <-
    "^(\\w+)/(\\w+)/(\\w+)/[A-Z]{2,3}_*[A-Z]*(adapt[0-9]|baseline[0-9]*).*"
  str_replace(mat, expr, replacement = "\\1 \\2 \\3 \\4") %>%
    str_split(pattern = " ", simplify = TRUE) %>% c()
}
get_feat_atyp <- function(mat) {
  expr <-
    "^(\\w+)/(\\w+)/(\\w+)/[A-Zb-z]{2,3}_{0,1}[0-9]*_{0,1}(\\S*)__MAT.mat"
  str_replace(mat, expr, "\\1 \\2 \\3 \\4") %>%
    str_split(" ", simplify = TRUE) %>% c()
}
feat_typ  <- map(fp_typ,  get_feat_typ)  %>% do.call(what = rbind)
feat_atyp <- map(fp_atyp, get_feat_atyp) %>% do.call(what = rbind)
cn <- c("task", "age_group", "initials", "trial")
colnames(feat_typ)  <- cn
colnames(feat_atyp) <- cn
feat_typ <- as_tibble(feat_typ) %>%
  add_column(path = fp_typ, .before = 1) %>%
  mutate(trial = if_else(str_detect(trial, "baseline"), "baseline", trial))
feat_atyp <- as_tibble(feat_atyp) %>%
  add_column(path = fp_atyp, .before = 1) %>%
  mutate(trial = if_else(trial == "", "baseline", trial)) %>%
  mutate(trial = str_replace(trial, "([A-Za-z]+)_*([0-9])+", "adapt\\2")) %>%
  mutate(trial = if_else(str_detect(trial, "adapt"), trial, "baseline"))
feature_tbl <- bind_rows(feat_typ, feat_atyp) %>%
  mutate(
    task = tolower(task),
    age_group = factor(age_group),
    age_group = recode_factor(
      age_group,
      Young = "young_adult",
      MiddleAge = "middle_age",
      Older = "older_adult",
    ),
    age_group = as.character(age_group)
  )
age_sex_tbl <- readxl::read_xlsx(
  "data/RecalibrationData/ParticipantAgeSex.xlsx",
  col_types = c("guess", "numeric", "guess")
) %>%
  mutate_at(vars(Sex), as.factor) %>%
  rename(initials = ID,
         age = Age,
         sex = Sex) %>%
  arrange(initials)
get_age_group <- function(age) {
  age_group <- vector("character", length(age))
  age_group[age >= 18 & age <= 30] <- "young_adult"
  age_group[age >= 39 & age <= 50] <- "middle_age"
  age_group[age >= 65 & age <= 75] <- "older_adult"
  age_group[age_group == ""] = NA
  age_group
}
feature_tbl <- feature_tbl %>%
  unite(col = "initials_age_group",
        initials,
        age_group,
        sep = "-",
        remove = TRUE)
age_sex_tbl <- age_sex_tbl %>%
  mutate(age_group = get_age_group(age)) %>%
  unite(col = "initials_age_group",
        initials,
        age_group,
        sep = "-",
        remove = TRUE)
feature_tbl <-
  full_join(feature_tbl, age_sex_tbl, by = "initials_age_group") %>%
  separate(initials_age_group, c("initials", "age_group"), sep = "-") %>%
  mutate(age_group = factor(
    age_group,
    levels = c("young_adult", "middle_age", "older_adult"),
    ordered = FALSE
  )) %>%
  mutate(trial = factor(
    trial,
    levels = c("baseline", "adapt1", "adapt2", "adapt3"),
    ordered = FALSE
  ))
features <- feature_tbl %>%
  mutate(
    tmp_task = recode_factor(
      task,
      audiovisual = "av",
      visual = "vis",
      sensorimotor = "sm",
      duration = "dur"
    ),
    trial = recode_factor(
      trial,
      baseline = "pre",
      adapt1 = "post1",
      adapt2 = "post2",
      adapt3 = "post3"
    ),
    initials = str_replace(initials, "JM_F", "JM"),
    tmp_sex = tolower(as.character(sex)),
    tmp_age_group = recode_factor(
      age_group,
      young_adult = "Y",
      middle_age = "M",
      older_adult = "O"
    )
  ) %>%
  unite(
    rid,
    tmp_task,
    trial,
    tmp_age_group,
    tmp_sex,
    initials,
    sep = "-",
    remove = FALSE
  ) %>%
  unite(sid,
        tmp_age_group,
        tmp_sex,
        initials,
        sep = "-",
        remove = FALSE) %>%
  select(-c(tmp_task, tmp_sex, tmp_age_group, initials)) %>%
  select(rid, sid, path, task, trial, age_group, age, sex) %>%
  arrange(rid) %>%
  mutate(rid = factor(rid),
         sid = factor(sid))
