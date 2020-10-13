library(tidyverse)
library(keras)
library(recipes)
library(tfdatasets)
library(rsample)

# Download data
claims <- cellar::cellar_pull("nfip_claims") %>%
  filter(year_of_loss > 2000)

# Basic feature engineering
small_data <- claims %>%
  filter(amount_paid_on_building_claim > 0,
         total_building_insurance_coverage > 0) %>%
  sample_n(10000) %>%
  select(amount_paid_on_building_claim, total_building_insurance_coverage,
         reported_zip_code, state, primary_residence, basement_enclosure_crawlspace_type,
         condominium_indicator, number_of_floors_in_the_insured_building, occupancy_type,
         community_rating_system_discount, flood_zone) %>%
  mutate(loss_proportion = pmin(amount_paid_on_building_claim / total_building_insurance_coverage, 1))

predictors <- c("total_building_insurance_coverage", "reported_zip_code",
                "primary_residence", "basement_enclosure_crawlspace_type",
                "number_of_floors_in_the_insured_building", "occupancy_type", "flood_zone",
                "community_rating_system_discount")
response <- "loss_proportion"

cvfolds <- small_data %>%
  rsample::vfold_cv()

nn_analyze_assess <- function(splits, ...) {
   analysis_data <- analysis(splits)
   assessment_data <- assessment(splits)

   spec <- feature_spec(analysis_data,
                        x = !!predictors,
                        y = !!response) %>%
     step_numeric_column(total_building_insurance_coverage,
                         community_rating_system_discount, normalizer_fn = scaler_standard()) %>%
     step_categorical_column_with_vocabulary_list(basement_enclosure_crawlspace_type,
                                                  number_of_floors_in_the_insured_building, occupancy_type,
                                                  flood_zone, primary_residence) %>%
     step_embedding_column(basement_enclosure_crawlspace_type,
                           number_of_floors_in_the_insured_building, occupancy_type,
                           dimension = 1) %>%
     step_embedding_column(flood_zone, dimension = 8) %>%
     step_indicator_column(primary_residence)

   spec <- fit(spec)

   features_layer <- layer_dense_features(feature_columns = dense_features(spec))
   input <- layer_input_from_dataset(analysis_data %>% select(.env$predictors))
   output <- input %>%
     features_layer() %>%
     layer_dense(units = 1, activation = "softplus")

   model <- keras_model(input, output)

   model %>%
     compile(loss = "mean_squared_error",
             optimizer = optimizer_adam(lr = 0.1),
             metrics = "mean_squared_error")

   history <- model %>%
     fit(x = analysis_data %>% select(.env$predictors),
         y = analysis_data$amount_paid_on_building_claim,
         validation_split = 0.2,
         epochs = 200,
         verbose = 0,
         callbacks = callback_early_stopping(restore_best_weights = TRUE))

   preds <- model %>%
     predict(assessment_data) %>%
      as.vector()

   actuals <- assessment_data %>%
      pull(amount_paid_on_building_claim)

   form <- amount_paid_on_building_claim ~ total_building_insurance_coverage +
      basement_enclosure_crawlspace_type + number_of_floors_in_the_insured_building +
      occupancy_type + flood_zone + primary_residence
   rec <- recipe(form,
                 data = analysis_data) %>%
      step_mutate(flood_zone = substr(flood_zone, 1, 1)) %>%
      step_unknown(occupancy_type, number_of_floors_in_the_insured_building,
                   flood_zone, primary_residence) %>%
      step_log(total_building_insurance_coverage) %>%
      prep()

   model_glm <- glm(form, family = Gamma(link = log), data = juice(rec),
               control = list(maxit = 100))

   preds_glm <- predict(model_glm, bake(rec, assessment_data), type = "response")

   list(
      actuals = actuals,
      preds_nn = preds,
      preds_glm = preds_glm
   )
}

cv_results <- cvfolds$splits[1:2] %>%
  map(nn_analyze_assess)

cv_results %>%
   map(function(x) {
      list(
         rmse_nn = sqrt(sum((x$actuals - x$preds_nn)^2) / length(x$actuals)),
         rmse_glm = sqrt(sum((x$actuals - x$preds_glm)^2) / length(x$actuals))
      )
   })
