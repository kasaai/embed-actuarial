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

env <- new.env()

cvfolds <- small_data %>%
    rsample::vfold_cv(v = 2)
cvfolds


model_analyze_assess <- function(splits, ...) {
    analysis_data <- analysis(splits)
    assessment_data <- assessment(splits)

    rec <- recipe(amount_paid_on_building_claim ~ .,
        data = analysis_data %>%
            select(-loss_proportion, -reported_zip_code)
    ) %>%
        step_normalize(all_numeric(), -all_outcomes()) %>%
        step_unknown(all_nominal()) %>%
        step_integer(all_nominal(), strict = TRUE, zero_based = TRUE) %>%
        prep()

    ds <- flood_dataset(juice(rec), categorical_cols, numeric_cols, response_col, env)
    test_ds <- flood_dataset(bake(rec, assessment_data), categorical_cols, numeric_cols)
    train_indx <- sample(length(ds), 0.8 * length(ds))
    valid_indx <- seq_len(length(ds)) %>% setdiff(train_indx)
    train_ds <- dataset_subset(ds, train_indx)
    valid_ds <- dataset_subset(ds, valid_indx)
    model <- net(env$cardinalities, length(numeric_cols), function(x) 2)

    train_dl <- train_ds %>% dataloader(batch_size = 1024, shuffle = TRUE)
    valid_dl <- valid_ds %>% dataloader(batch_size = 1024, shuffle = TRUE)
    test_dl <- test_ds %>% dataloader(batch_size = 1024, shuffle = FALSE)

    model <- net(env$cardinalities, length(numeric_cols), function(x) 2)

    optimizer <- optim_adam(model$parameters, lr = 0.1)

    train_loop(model, train_dl, valid_dl, 2, optimizer)

    preds_nn <- get_preds(model, test_dl)

    form <- amount_paid_on_building_claim ~ total_building_insurance_coverage +
        basement_enclosure_crawlspace_type + number_of_floors_in_the_insured_building +
        occupancy_type + flood_zone + primary_residence

    rec <- recipe(form,
        data = analysis_data
    ) %>%
        step_mutate(flood_zone = substr(flood_zone, 1, 1)) %>%
        step_unknown(
            occupancy_type, number_of_floors_in_the_insured_building,
            flood_zone, primary_residence
        ) %>%
        step_log(total_building_insurance_coverage) %>%
        prep()

    model_glm <- glm(form,
        family = Gamma(link = log), data = juice(rec),
        control = list(maxit = 100)
    )

    preds_glm <- predict(model_glm, bake(rec, assessment_data), type = "response")

    actuals <- assessment_data %>%
        pull(amount_paid_on_building_claim)

    list(
        actuals = actuals,
        preds_nn = preds_nn,
        preds_glm = preds_glm
    )
}

cv_results <- cvfolds$splits %>%
    map(model_analyze_assess)

cv_results %>%
    map(function(x) {
        list(
            #  rmse_nn = sqrt(sum((x$actuals - x$preds_nn)^2) / length(x$actuals)),
            rmse_glm = sqrt(sum((x$actuals - x$preds_glm)^2) / length(x$actuals))
        )
    })
