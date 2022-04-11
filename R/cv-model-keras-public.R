library(tidyverse)
library(recipes)
library(rsample)
library(torch)
library(data.table)
library(keras)
require(tensorflow)
require(tabnet)
tf$compat$v1 = T

source("model-utils.R")
source("data-loading.R")

source("keras_working_file.R")
source("keras_working_file_attn.R")
source("keras_working_file_tabtrans.R")

categorical_cols <- c(
  "primary_residence", "basement_enclosure_crawlspace_type",
  "number_of_floors_in_the_insured_building",
  "occupancy_type", "flood_zone"
)

numeric_cols <- c("total_building_insurance_coverage", "community_rating_system_discount")
response_col <- "amount_paid_on_building_claim"

########### Choose dataset

### for finetuning use first 100k
#for_cv = small_data[1:100000,]

### for test use second 100k
for_test = small_data[100000:200000,]

set.seed(420)
cvfolds <- for_cv %>%
  rsample::vfold_cv(v = 5)

# ### code to test analysis function; comment out for full CV
#
# splits = cvfolds$splits[[1]]
#
# learning_rate = 0.001
# epochs = 10
# batch_size = 5000

### Run all models

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
    step_integer(all_nominal(), strict = TRUE, zero_based = FALSE) %>%
    prep(strings_as_factors = TRUE)

  juiced_train_data <- juice(rec_nn)
  baked_test_data <- bake(rec_nn, assessment_data)
  for_card = juiced_train_data %>% data.table()
  for_card_test = baked_test_data %>% data.table()

  output_list = get_output(for_card, for_card_test)

  train_dl = juiced_train_data %>% get_keras_data()
  test_dl = baked_test_data %>% get_keras_data()
  actuals <- assessment_data %>%
    pull(amount_paid_on_building_claim)

  # Model02

  simple_nn = get_keras_model(hidden_layer_units = 64,hidden_layer_activations = "tanh", embed = 1, for_card, train_dl, test_dl, output_list)
  preds_nn <- simple_nn$test_preds

  # Model04

  simple_nn_multidim = get_keras_model(hidden_layer_units = 64,hidden_layer_activations = "tanh", embed = "multi" ,for_card, train_dl, test_dl, output_list)
  preds_nn2 <- simple_nn_multidim$test_preds

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
                             data = juice(rec_tabnet), epochs = 10,
                             num_independent = 1, num_shared = 1, num_steps = 3,
                             decision_width = 1, attention_width = 1,
                             learn_rate = 1, verbose = TRUE, batch_size = 2000,
                             valid_split = 0.2
  )

  preds_tabnet <- predict(model_tabnet, bake(rec_tabnet, assessment_data))$.pred

  ### basic GLM - gamma

  rec_glm <- recipe(form,
                    data = analysis_data) %>%
    step_mutate(flood_zone = substr(flood_zone, 1, 1)) %>%
    step_novel(all_nominal()) %>%
    step_log(total_building_insurance_coverage) %>%
    prep(strings_as_factors = FALSE)

  model_glm <- glm(form,
                   family = Gamma(link = log), data = juice(rec_glm),
                   control = list(maxit = 100)
  )

  preds_glm <- predict(model_glm, bake(rec_glm, assessment_data), type = "response")

  ### basic GLM - Gaussian

  model_glm_gaussian <- glm(form, family = gaussian, data = juice(rec_glm), control = list(maxit = 100))

  preds_glm_gaussian <- predict(model_glm_gaussian, bake(rec_glm, assessment_data), type = "response")

  ### glm - replace with unidimensional embeddings

  glm_embed = get_glm_embed(simple_nn$model, for_card, train_dl, test_dl, output_list)
  glm_transfer_train = glm_embed$glm_train
  glm_transfer_train[, V6 := (juice(rec_glm))$total_building_insurance_coverage]
  glm_transfer_test = glm_embed$glm_test
  glm_transfer_test[, V6 := (bake(rec_glm, assessment_data))$total_building_insurance_coverage]

  model_glm_transfer <- glm(amount_paid_on_building_claim~.,
                            family = Gamma(link = log), data = glm_transfer_train,
                            control = list(maxit = 500))

  preds_glm_train <- predict(model_glm_transfer, glm_transfer_train, type = "response")
  preds_glm_test <- predict(model_glm_transfer, glm_transfer_test, type = "response")

  preds_glm2 <- preds_glm_test

  ### Attention models
  simple_attn = get_keras_model_attn(hidden_layer_units = 64, embed = 0, for_card, train_dl, test_dl, output_list)
  preds_simple_attn <- simple_attn$test_preds

  tabtrans = get_keras_model_tabtrans(hidden_layer_units = 64, embed = 0, for_card, train_dl, test_dl, output_list)
  preds_tabt <- tabtrans$test_preds

  list(

    model_nn = simple_nn$model,
    model_nn2 = simple_nn_multidim$model,
    model_simple_attn = simple_attn$model,
    model_tabt = tabtrans$model,
    glm_simple = model_glm,
    model_glm_gaussian=model_glm_gaussian,
    model_glm_transfer=model_glm_transfer,
    rec_nn = rec_nn,
    actuals = actuals,
    preds_nn = preds_nn,
    preds_nn2 = preds_nn2,
    preds_glm = preds_glm,
    preds_glm2 = preds_glm2,
    preds_glm_gaussian = preds_glm_gaussian,
    preds_simple_attn = preds_simple_attn,
    preds_tabt = preds_tabt,
    preds_tabnet = preds_tabnet

  )
}

cv_results <- cvfolds$splits %>%
  lapply(function(x) model_analyze_assess(x, 0.001, 30, 1024))

cv_results %>% save(file = "c:/r/test_set.rda")

require(data.table)
res = cv_results %>%
  map(function(x) {
    list(
      rmse_nn = rmse(x$actuals, x$preds_nn),
      rmse_nn2 = rmse(x$actuals, x$preds_nn2),
      rmse_glm = rmse(x$actuals, x$preds_glm),
      rmse_glm2 = rmse(x$actuals, x$preds_glm2),
      rmse_glm_gaussian = rmse(x$actuals, x$preds_glm_gaussian),
      rmse_simple_attn = rmse(x$actuals, x$preds_simple_attn),
      rmse_tabt = rmse(x$actuals, x$preds_tabt),
      rmse_tabnet = rmse(x$actuals, x$preds_tabnet),
      mae_nn = mae(x$actuals, x$preds_nn),
      mae_nn2 = mae(x$actuals, x$preds_nn2),
      mae_glm = mae(x$actuals, x$preds_glm),
      mae_glm2 = mae(x$actuals, x$preds_glm2),
      mae_glm_gaussian = mae(x$actuals, x$preds_glm_gaussian),
      mae_simpleattn = mae(x$actuals, x$preds_simple_attn),
      mae_tabt = mae(x$actuals, x$preds_tabt),
      mae_tabnet = mae(x$actuals, x$preds_tabnet)
    )
  }) %>% rbindlist()

res[, id:= 1:5]
res[, run := "final"]
res %>% fwrite("c:/r/final.csv")

all_res = res %>% melt.data.table(id.vars = c("run", "id"))
melted = all_res[, mean(value), keyby = .(run, variable)]
melted = melted[order(run)]%>%
  dcast.data.table(variable~run)

melted %>% fwrite("c:/r/final_means.csv")
