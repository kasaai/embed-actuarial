library(tidyverse)
library(recipes)
library(rsample)
library(torch)
library(tabnet)

source("R/model-utils.R")
source("R/data-loading.R")

categorical_cols <- c(
  "primary_residence", "basement_enclosure_crawlspace_type",
  "number_of_floors_in_the_insured_building",
  "occupancy_type", "flood_zone"
)

numeric_cols <- c("total_building_insurance_coverage", "community_rating_system_discount")
response_col <- "amount_paid_on_building_claim"
coverage_col <- "coverage"

set.seed(420)
cvfolds <- small_data %>%
  rsample::vfold_cv(v = 5)

####### get data
splits = cvfolds$splits[[1]]
env <- new.env()
analysis_data <- analysis(splits)
assessment_data <- assessment(splits)

batch_size = 1

rec_nn <- recipe(amount_paid_on_building_claim ~ .,
                 data = analysis_data %>%
                   select(-loss_proportion, -reported_zip_code)
) %>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_novel(all_nominal()) %>%
  step_integer(all_nominal(), strict = TRUE, zero_based = TRUE) %>%
  prep(strings_as_factors = TRUE)

ds <- flood_dataset(
  juice(rec_nn) %>% mutate(coverage = analysis_data$total_building_insurance_coverage),
  categorical_cols, numeric_cols, coverage_col, response_col, env
)
baked_test_data <- bake(rec_nn, assessment_data)

test_ds <- flood_dataset(
  baked_test_data %>% mutate(coverage = assessment_data$total_building_insurance_coverage),
  categorical_cols, numeric_cols, coverage_col
)

train_indx <- sample(length(ds), 0.8 * length(ds))
valid_indx <- seq_len(length(ds)) %>% setdiff(train_indx)
train_ds <- dataset_subset(ds, train_indx)
valid_ds <- dataset_subset(ds, valid_indx)

train_dl <- train_ds %>% dataloader(batch_size = batch_size, shuffle = TRUE)
valid_dl <- valid_ds %>% dataloader(batch_size = batch_size, shuffle = TRUE)
test_dl <- test_ds %>% dataloader(batch_size = batch_size, shuffle = FALSE)
####################

xcat = train_ds[20]$x[[1]]$to(device = "cpu")
xcat = torch_reshape(xcat, c(1,5))

#### simple attn

model <- torch_load("./model_files/simple_attn.pt")


embedded <- model$embedder(xcat)
shapes <- embedded$shape
embedded_reshape <- embedded$view(list(model$embed_dim, embedded$shape[1], model$embed_dim))
embedded_attention <- model$attn(embedded_reshape, embedded_reshape, embedded_reshape)
embedded_attended <- embedded_attention[[1]]
embedded_attended <- embedded_attended$view(list(embedded$shape[1], model$embed_dim * model$embed_dim))
embedded_attended_reshape = torch_reshape(embedded_attended, c(1,5,5))
embedded_attention[[2]]

#### tab transformer
model <- torch_load("./model_files/tabt.pt")
xcat_out <- model$col_embedder(xcat)
shapes <- xcat_out$shape
embedded_reshape <- xcat_out$view(list(shapes[2], shapes[1], shapes[3]))

attn = model$attn(embedded_reshape, embedded_reshape, embedded_reshape)[[2]]
embedded_reshape <- model$attn(embedded_reshape, embedded_reshape, embedded_reshape)[[1]] + embedded_reshape
embedded_reshape <- model$lnorm1(embedded_reshape)
xcat_out_a <- embedded_reshape %>%
  model$linear1() %>%
  nnf_relu() %>%
  model$linear2()
embedded_reshape <- model$lnorm2(embedded_reshape + xcat_out_a)
embedded_reshape <- embedded_reshape$view(list(shapes[1], shapes[2]*shapes[3]))
concat <- torch_cat(list(embedded_reshape, xnum), dim = 2)


level_integer_mappings <- rec_nn$steps[[3]]$key %>%
  lapply(function(x) mutate(x, integer = as.integer(integer + 1L)))

embeddings <- model$embedder$embeddings %>%
  (function(x) lapply(1:length(categorical_cols), function(i) x[[i]])) %>%
  setNames(categorical_cols) %>%
  purrr::imap(function(m, v) {
    outputs <- level_integer_mappings[[v]]$integer %>%
      torch_tensor(device = "cpu") %>%
      m() %>%
      (function(x) x$to(device = "cpu")) %>%
      as.array()
    mapping_tbl <- level_integer_mappings[[v]]
    cbind(
      level = mapping_tbl$value,
      as.data.frame(outputs) %>% setNames(paste0("e", 1:ncol(outputs)))
    )
  })


emb_flood_zone <- embeddings$flood_zone %>%
  rename(embedding = e1)
list(
  emb_flood_zone[1:10, ], emb_flood_zone[11:20, ],
  emb_flood_zone[21:30, ], emb_flood_zone[31:40, ],
  emb_flood_zone[41:50, ]
) %>%
  Reduce(cbind, .) %>%
  knitr::kable(digits = 2)

emb_occupancy_type <- embeddings$occupancy_type %>%
  rename(embedding = e1) %>%
  filter(!level == "new")
knitr::kable(emb_occupancy_type, digits = 2)

# for multidimensional

emb_flood_zone <- embeddings$flood_zone %>%
  filter(level == "A00") %>%
  select(-level) %>%
  tidyr::pivot_longer(starts_with("e"), names_to = "dimension") %>%
  mutate(value = format(value, digits = 2))

# withr::with_options(list(knitr.kable.NA = ""), {

# })

list(
  emb_flood_zone[1:10, ], emb_flood_zone[11:20, ],
  emb_flood_zone[21:30, ],
  rbind(
    emb_flood_zone[31:34, ],
    data.frame(dimension = rep("", 6), value = rep("", 6))
  )
) %>%
  # dplyr::bind_cols() %>%
  Reduce(cbind, .) %>%
  knitr::kable(digits = 2, format = "pipe")



emb_occupancy_type <- embeddings$occupancy_type %>%
  filter(!level == "new")
knitr::kable(emb_occupancy_type, digits = 2, format = "pipe")

embeddings$number_of_floors_in_the_insured_building
embeddings %>% names()
p <- embeddings$number_of_floors_in_the_insured_building %>%
  filter(level != "new") %>%
  ggplot() +
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_text(aes(-0.5, e1, label = round(e1, 2))) +
  geom_text(aes(1, e1, label = stringr::str_wrap(level, 40)), hjust = 0) +
  geom_point(aes(0, e1)) +
  geom_segment(aes(x = 0.8, xend = 0.1, y = e1, yend = e1),
               arrow = arrow(type = "closed", length = unit(0.075, "inches")),
               alpha = 0.5
  ) +
  coord_cartesian(xlim = c(-3, 8)) +
  coord_flip() +
  theme_void()
p2 <- embeddings$number_of_floors_in_the_insured_building %>%
  filter(level != "new") %>%
  ggplot() +
  coord_cartesian(ylim = c(-1, 1), xlim = c(NA, -1)) +
  # coord_cartesian(ylim = c(-3, 3), xlim = c(-5, 3)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_text(aes(e1, 0.1, label = round(e1, 2), angle = 45)) +
  geom_text(aes(e1, -0.1, label = stringr::str_wrap(level, 40)), hjust = 0, angle = -45) +
  geom_point(aes(e1, 0)) +
  theme_void()
# geom_segment(aes(x = 0.8, xend = 0.1, y = e1, yend = e1),
#     arrow = arrow(type = "closed", length = unit(0.075, "inches")),
#     alpha = 0.5
# ) +
# coord_cartesian(xlim = c(-3, 8)) +
#     theme_void()
p2
ggsave("manuscript/images/num_floors.png", plot = p2, width = 8, height = 5)

library(ggrepel)
p3 <- embeddings$flood_zone %>%
  filter(level != "new") %>%
  mutate(prefix = substr(level, 1, 1)) %>%
  ggplot(aes(e1, prefix, color = prefix, label = level)) +
  geom_point() +
  geom_label_repel() +
  theme_bw()

ggsave("manuscript/images/flood_zone.png", plot = p3, width = 8, height = 8)


embeddings$flood_zone %>%
  mutate(prefix = substr(level, 1, 1)) %>%
  ggplot() +
  geom_point(aes(e1, e2, color = prefix), alpha = 1)

embeddings$state %>%
  ggplot() +
  geom_point(aes(e1, e2)) +
  geom_text(aes(e1, e2, label = level))

embeddings$flood_zone %>% dim()


####### PCA stuff


embeddings$flood_zone %>% str()

lbls <- embeddings$flood_zone$level
wts <- embeddings$flood_zone %>%
  select(starts_with("e")) %>%
  as.matrix()

pca <- prcomp(wts, center = TRUE, scale. = TRUE, rank = 2)$x[, c("PC1", "PC2")] %>%
  as.data.frame()


ggplot(data = pca, aes(x = PC1, y = PC2)) +
  geom_point()
library(ggrepel)
p4 <- pca %>%
  as.data.frame() %>%
  mutate(class = lbls) %>%
  mutate(prefix = substr(lbls, 1, 1)) %>%
  filter(class != "new") %>%
  ggplot(aes(x = PC1, y = PC2, color = prefix)) +
  geom_point() +
  geom_label_repel(aes(label = class)) +
  # coord_cartesian(xlim = c(-2, 2), ylim = c(-2, 2)) +
  # theme(aspect.ratio = 1) +
  theme_classic()

ggsave("manuscript/images/flood_zone_pca.png", plot = p4)

library(Rtsne)
tsne_out <- Rtsne::Rtsne(wts, perplexity = 5)
p5 <- tsne_out$Y %>%
  as.data.frame() %>%
  mutate(class = lbls) %>%
  mutate(prefix = substr(lbls, 1, 1)) %>%
  filter(class != "new") %>%
  ggplot(aes(x = V1, y = V2, color = prefix)) +
  geom_point() +
  geom_label_repel(aes(label = class)) +
  theme_classic()

ggsave("manuscript/images/flood_zone_tsne.png", plot = p5)
