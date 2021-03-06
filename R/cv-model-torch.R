library(tidyverse)
library(recipes)
library(rsample)
library(torch)

source("R/model-utils.R")
source("R/data-loading.R")

categorical_cols <- c(
    "state", "primary_residence", "basement_enclosure_crawlspace_type",
    "condominium_indicator", "number_of_floors_in_the_insured_building",
    "occupancy_type", "flood_zone"
)

numeric_cols <- c("total_building_insurance_coverage", "community_rating_system_discount")
response_col <- "amount_paid_on_building_claim"


cvfolds <- small_data %>%
    rsample::vfold_cv(v = 2)
cvfolds

model_analyze_assess <- function(splits, learning_rate = 1, epochs = 10, batch_size = 5000, ...) {
    env <- new.env()
    analysis_data <- analysis(splits)
    assessment_data <- assessment(splits)

    rec <- recipe(amount_paid_on_building_claim ~ .,
        data = analysis_data %>%
            select(-loss_proportion, -reported_zip_code)
    ) %>%
        step_normalize(all_numeric(), -all_outcomes()) %>%
        step_novel(all_nominal()) %>%
        # step_unknown(all_nominal(), new_level = "missing") %>%
        step_integer(all_nominal(), strict = TRUE, zero_based = TRUE) %>%
        prep(strings_as_factors = TRUE)

    ds <- flood_dataset(juice(rec), categorical_cols, numeric_cols, response_col, env)
    baked_test_data <- bake(rec, assessment_data)
    test_ds <- flood_dataset(baked_test_data, categorical_cols, numeric_cols)
    train_indx <- sample(length(ds), 0.8 * length(ds))
    valid_indx <- seq_len(length(ds)) %>% setdiff(train_indx)
    train_ds <- dataset_subset(ds, train_indx)
    valid_ds <- dataset_subset(ds, valid_indx)

    train_dl <- train_ds %>% dataloader(batch_size = batch_size, shuffle = TRUE)
    valid_dl <- valid_ds %>% dataloader(batch_size = batch_size, shuffle = TRUE)
    test_dl <- test_ds %>% dataloader(batch_size = batch_size, shuffle = FALSE)

    model <- trivial_net(env$cardinalities, length(numeric_cols), function(x) 1)

    optimizer <- optim_adam(model$parameters, lr = learning_rate)

    train_loop(model, train_dl, valid_dl, epochs, optimizer)

    preds_nn <- get_preds(model, test_dl)

    form <- amount_paid_on_building_claim ~ total_building_insurance_coverage +
        basement_enclosure_crawlspace_type + number_of_floors_in_the_insured_building +
        occupancy_type + flood_zone + primary_residence

    rec <- recipe(form,
        data = analysis_data
    ) %>%
        step_mutate(flood_zone = substr(flood_zone, 1, 1)) %>%
        step_novel(all_nominal()) %>%
        # step_unknown(all_nominal(), new_level = "missing") %>%
        step_log(total_building_insurance_coverage) %>%
        prep(strings_as_factors = FALSE)

    model_glm <- glm(form,
        family = Gamma(link = log), data = juice(rec),
        control = list(maxit = 100)
    )

    preds_glm <- predict(model_glm, bake(rec, assessment_data), type = "response")

    actuals <- assessment_data %>%
        pull(amount_paid_on_building_claim)

    list(
        model_nn = model,
        model_glm = model_glm,
        actuals = actuals,
        preds_nn = preds_nn,
        preds_glm = preds_glm
    )
}

cv_results <- cvfolds$splits %>%
    lapply(function(x) model_analyze_assess(x, 0.01, 50, 10000))

cv_results %>%
    map(function(x) {
        list(
            rmse_nn = sqrt(sum((x$actuals - x$preds_nn)^2) / length(x$actuals)),
            rmse_glm = sqrt(sum((x$actuals - x$preds_glm)^2) / length(x$actuals))
        )
    })