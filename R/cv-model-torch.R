library(tidyverse)
library(recipes)
library(rsample)
library(torch)

source("R/model-utils.R")
source("R/data-loading.R")

categorical_cols <- c(
    "primary_residence", "basement_enclosure_crawlspace_type",
    "number_of_floors_in_the_insured_building",
    "occupancy_type", "flood_zone"
)

numeric_cols <- c("total_building_insurance_coverage", "community_rating_system_discount")
response_col <- "amount_paid_on_building_claim"


cvfolds <- small_data %>%
    rsample::vfold_cv(v = 5)
# cvfolds

model_analyze_assess <- function(splits, learning_rate = 1, epochs = 10, batch_size = 5000, ...) {
    env <- new.env()
    analysis_data <- analysis(splits)
    assessment_data <- assessment(splits)

    rec_nn <- recipe(amount_paid_on_building_claim ~ .,
        data = analysis_data %>%
            select(-loss_proportion, -reported_zip_code)
    ) %>%
        step_log(total_building_insurance_coverage) %>%
        step_normalize(all_numeric(), -all_outcomes()) %>%
        step_novel(all_nominal()) %>%
        # step_unknown(all_nominal(), new_level = "missing") %>%
        step_integer(all_nominal(), strict = TRUE, zero_based = TRUE) %>%
        prep(strings_as_factors = TRUE)

    ds <- flood_dataset(juice(rec_nn), categorical_cols, numeric_cols, response_col, env)
    baked_test_data <- bake(rec_nn, assessment_data)
    test_ds <- flood_dataset(baked_test_data, categorical_cols, numeric_cols)
    train_indx <- sample(length(ds), 0.8 * length(ds))
    valid_indx <- seq_len(length(ds)) %>% setdiff(train_indx)
    train_ds <- dataset_subset(ds, train_indx)
    valid_ds <- dataset_subset(ds, valid_indx)

    train_dl <- train_ds %>% dataloader(batch_size = batch_size, shuffle = TRUE)
    valid_dl <- valid_ds %>% dataloader(batch_size = batch_size, shuffle = TRUE)
    test_dl <- test_ds %>% dataloader(batch_size = batch_size, shuffle = FALSE)

    model <- simple_net(env$cardinalities, length(numeric_cols), units = 16, fn_embedding_dim = function(x) 1)

    optimizer <- optim_adam(model$parameters, lr = learning_rate, weight_decay = 0)
    # optimizer <- optim_rmsprop(model$parameters, lr = 0.1)

    train_loop(model, train_dl, valid_dl, epochs, optimizer)

    replace_unseen_level_weights_(model$embedder$embeddings)

    preds_nn <- get_preds(model, test_dl)

    model2 <- simple_net(env$cardinalities, length(numeric_cols),
        units = 16,
        fn_embedding_dim = function(x) ceiling(x / 2)
    )

    optimizer <- optim_adam(model2$parameters, lr = learning_rate, weight_decay = 0)

    train_loop(model2, train_dl, valid_dl, epochs, optimizer)

    replace_unseen_level_weights_(model2$embedder$embeddings)

    preds_nn2 <- get_preds(model2, test_dl)

    form <- amount_paid_on_building_claim ~ total_building_insurance_coverage +
        basement_enclosure_crawlspace_type + number_of_floors_in_the_insured_building +
        occupancy_type + flood_zone + primary_residence + community_rating_system_discount

    rec_glm <- recipe(form,
        data = analysis_data
    ) %>%
        step_mutate(flood_zone = substr(flood_zone, 1, 1)) %>%
        step_novel(all_nominal()) %>%
        # step_unknown(all_nominal(), new_level = "missing") %>%
        step_log(total_building_insurance_coverage) %>%
        prep(strings_as_factors = FALSE)

    model_glm <- glm(form,
        family = Gamma(link = log), data = juice(rec_glm),
        control = list(maxit = 100)
    )

    preds_glm <- predict(model_glm, bake(rec_glm, assessment_data), type = "response")

    rec_glm2 <- recipe(form,
        data = analysis_data
    ) %>%
        step_novel(all_nominal()) %>%
        step_log(total_building_insurance_coverage) %>%
        prep(strings_as_factors = FALSE)

    key <- key_with_embeddings(model$embedder$embeddings, rec_nn$steps[[4]]$key)
    model_glm2 <- glm(form,
        family = Gamma(link = log), data = map_cats_to_embeddings(juice(rec_glm2), key),
        control = list(maxit = 100)
    )
    preds_glm2 <- predict(model_glm2,
        bake(rec_glm2, assessment_data) %>% map_cats_to_embeddings(key),
        type = "response"
    )

    actuals <- assessment_data %>%
        pull(amount_paid_on_building_claim)

    list(
        model_nn = model,
        model_nn2 = model2,
        rec_nn = rec_nn,
        model_glm = model_glm,
        rec_glm = rec_glm,
        model_glm2 = model_glm2,
        rec_glm2 = rec_glm2,
        actuals = actuals,
        preds_nn = preds_nn,
        preds_nn2 = preds_nn2,
        preds_glm = preds_glm,
        preds_glm2 = preds_glm2
    )
}

cv_results <- cvfolds$splits %>%
    lapply(function(x) model_analyze_assess(x, 0.01, 1, 1100))

cv_results %>%
    map(function(x) {
        list(
            rmse_nn = sqrt(sum((x$actuals - x$preds_nn)^2) / length(x$actuals)),
            rmse_nn2 = sqrt(sum((x$actuals - x$preds_nn2)^2) / length(x$actuals)),
            rmse_glm = sqrt(sum((x$actuals - x$preds_glm)^2) / length(x$actuals)),
            rmse_glm2 = sqrt(sum((x$actuals - x$preds_glm2)^2) / length(x$actuals))
        )
    })
