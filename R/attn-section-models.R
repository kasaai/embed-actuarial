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

model_analyze_assess <- function(splits, learning_rate = 1, epochs = 10, batch_size = 5000, ...) {
  env <- new.env()
  analysis_data <- analysis(splits)
  assessment_data <- assessment(splits)

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

  # tabtransformer
  model_tabt <- tabtransformer(env$cardinalities, length(numeric_cols),
                               embedding_dim = 8, num_heads = 3, fc_units = 8
  )

  optimizer <- optim_adam(model_tabt$parameters, lr = learning_rate)

  train_loop(model_tabt, train_dl, valid_dl, epochs, optimizer, patience = 5, "tabt")

  model_tabt = torch_load("model_files/tabt.pt")

  replace_unseen_level_weights_(model_tabt$col_embedder$embeddings)

  preds_tabt <- get_preds(model_tabt, test_dl)


  # simple attention

  model_simple_attn <- simple_net_attn(env$cardinalities, length(numeric_cols),
                                       units = 8)

  optimizer <- optim_adam(model_simple_attn$parameters, lr = learning_rate, amsgrad = TRUE)

  train_loop(model_simple_attn, train_dl, valid_dl, epochs, optimizer, patience = 5, "simple_attn")

  model_simple_attn = torch_load("model_files/simple_attn.pt")

  replace_unseen_level_weights_(model_simple_attn$embedder$embeddings)

  preds_simple_attn <- get_preds(model_simple_attn, test_dl)


  # formula used for both tabnet and glm
  form <- amount_paid_on_building_claim ~ total_building_insurance_coverage +
    basement_enclosure_crawlspace_type + number_of_floors_in_the_insured_building +
    occupancy_type + flood_zone + primary_residence + community_rating_system_discount

  # tabnet
  rec_tabnet <- recipe(amount_paid_on_building_claim ~ .,
                       data = analysis_data %>%
                         select(-loss_proportion, -reported_zip_code)
  ) %>%
    step_normalize(all_numeric(), -all_outcomes()) %>%
    step_novel(all_nominal()) %>%
    prep(strings_as_factors = TRUE)

  model_tabnet <- tabnet_fit(form,
                             data = juice(rec_tabnet), epochs = epochs,
                             verbose = TRUE,
                             valid_split = 0.2
  )

  preds_tabnet <- predict(model_tabnet, bake(rec_tabnet, assessment_data))$.pred



  actuals <- assessment_data %>%
    pull(amount_paid_on_building_claim)

  list(
    model_tabnet = model_tabt,
    model_tabt = model_tabnet,
    model_simple_attn = model_simple_attn,
    rec_nn = rec_nn,
    actuals = actuals,
    preds_tabnet = preds_tabnet,
    preds_simple_attn = preds_simple_attn,
    preds_tabt = preds_tabt
  )
}

cv_results <- cvfolds$splits %>%
  lapply(function(x) model_analyze_assess(x, 0.001, 30, 1000))

model_names_mapping = data.frame(mod = c("tabnet", "tabt", "simpleattn"), Model = c("TabNet", "TabTransformer", "Simple Attention"))

require(data.table)
res = cv_results %>%
  map(function(x) {
    list(
      rmse_tabnet = rmse(x$actuals, pmax(x$preds_tabnet, 0.01)),
      rmse_tabt = rmse(x$actuals, pmax(x$preds_tabt, 0.01)),
      rmse_simpleattn = rmse(x$actuals, x$preds_simple_attn),
      mae_tabnet = mae(x$actuals, x$preds_tabnet),
      mae_tabt = mae(x$actuals, x$preds_tabt),
      mae_simpleattn = mae(x$actuals, x$preds_simple_attn)
    )
  }) %>% rbindlist()
res[, id:= 1:5]
res = res %>% melt.data.table(id.vars = c("id"))
res = res %>% separate(variable, into = c("metric", "mod"), sep = "_", extra = "merge") %>%
pivot_wider(names_from = "metric") %>%
  left_join(model_names_mapping, by = "mod") %>%
  select(-mod) %>% data.table()
res[, .(RMSE = mean(rmse), MAE = mean(mae)), keyby = .(Model)] %>%
  knitr::kable(digits = 0, format.args = list(
    big.mark = ",",
    scientific = FALSE
  ), format = "pipe")




_____________________________________________________

dir.create("model_files")
saveRDS(cv_results[[1]]$model_glm, "model_files/glm1.rds")
saveRDS(cv_results[[1]]$model_glm2, "model_files/glm2.rds")
saveRDS(cv_results[[1]]$rec_nn$steps[[3]]$key, "model_files/rec_nn_key.rds")
torch_save(cv_results[[1]]$model_nn, "model_files/nn1.pt")
torch_save(cv_results[[1]]$model_nn2, "model_files/nn2.pt")
torch_save(cv_results[[1]]$model_simple_attn, "model_files/nn_simple_attn.pt")
