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

    # Model02
    model <- simple_net(
        env$cardinalities, length(numeric_cols),
        units = 16, fn_embedding_dim = function(x) 1
    )

    # model <- model$to(device = "cuda")

    optimizer <- optim_adam(model$parameters, lr = learning_rate, amsgrad = TRUE)

    train_loop(model, train_dl, valid_dl, epochs, optimizer)

    replace_unseen_level_weights_(model$embedder$embeddings)

    preds_nn <- get_preds(model, test_dl)

    # Model04
    model2 <- simple_net(env$cardinalities, length(numeric_cols),
        units = 16,
        fn_embedding_dim = function(x) ceiling(x / 2)
    )

    # model2 <- model2$to(device = "cuda")

    optimizer <- optim_adam(model2$parameters, lr = learning_rate, amsgrad = TRUE)

    train_loop(model2, train_dl, valid_dl, epochs, optimizer)

    replace_unseen_level_weights_(model2$embedder$embeddings)

    preds_nn2 <- get_preds(model2, test_dl)

    # formula used for both tabnet and glm
    form <- amount_paid_on_building_claim ~ total_building_insurance_coverage +
        basement_enclosure_crawlspace_type + number_of_floors_in_the_insured_building +
        occupancy_type + flood_zone + primary_residence + community_rating_system_discount

    rec_glm <- recipe(form,
        data = analysis_data
    ) %>%
        step_mutate(flood_zone = substr(flood_zone, 1, 1)) %>%
        step_novel(all_nominal()) %>%
        step_log(total_building_insurance_coverage) %>%
        prep(strings_as_factors = FALSE)

    model_glm <- glm(form,
        family = Gamma(link = log), data = juice(rec_glm),
        control = list(maxit = 100)
    )

    preds_glm <- predict(model_glm, bake(rec_glm, assessment_data), type = "response")

    model_glm_gaussian <- glm(form, family = gaussian, data = juice(rec_glm), control = list(maxit = 100))

    preds_glm_gaussian <- predict(model_glm_gaussian, bake(rec_glm, assessment_data), type = "response")

    rec_glm2 <- recipe(form,
        data = analysis_data
    ) %>%
        step_novel(all_nominal()) %>%
        step_log(total_building_insurance_coverage) %>%
        prep(strings_as_factors = FALSE)

    key <- key_with_embeddings(model$embedder$embeddings, rec_nn$steps[[3]]$key)
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
        model_glm_gaussian = model_glm_gaussian,
        model_glm2 = model_glm2,
        rec_glm2 = rec_glm2,
        actuals = actuals,
        preds_nn = preds_nn,
        preds_nn2 = preds_nn2,
        preds_glm = preds_glm,
        preds_glm_gaussian = preds_glm_gaussian,
        preds_glm2 = preds_glm2
    )
}

cv_results <- cvfolds$splits %>%
    lapply(function(x) model_analyze_assess(x, 0.01, 15, 1000))

cv_results %>%
    map(function(x) {
        list(
            rmse_nn = rmse(x$actuals, pmax(x$preds_nn, 0.01)),
            rmse_nn2 = rmse(x$actuals, pmax(x$preds_nn2, 0.01)),
            rmse_glm = rmse(x$actuals, x$preds_glm),
            rmse_glm2 = rmse(x$actuals, x$preds_glm2),
            rmse_glm_gaussian = rmse(x$actuals, pmax(x$preds_glm_gaussian, 0.01)),
            rmse_zeros = rmse(x$actuals, 0),
            mae_nn = mae(x$actuals, x$preds_nn),
            mae_nn2 = mae(x$actuals, x$preds_nn2),
            mae_glm = mae(x$actuals, x$preds_glm),
            mae_glm2 = mae(x$actuals, x$preds_glm2),
            mae_glm_gaussian = mae(x$actuals, pmax(x$preds_glm_gaussian, 0.01))
        )
    }) %>%
    transpose() %>%
    as_tibble() %>%
    pivot_longer(everything()) %>%
    separate(name, into = c("metric", "mod"), sep = "_", extra = "merge") %>%
    pivot_wider(names_from = "metric") %>%
    left_join(model_names_mapping, by = "mod") %>%
    select(-mod) %>%
    arrange(id) %>%
    select(Model, rmse, mae) %>%
    rename(RMSE = rmse, MAE = mae) %>%
    knitr::kable(digits = 0, format.args = list(
        big.mark = ",",
        scientific = FALSE
    ), format = "pipe")

dir.create("model_files")
saveRDS(cv_results[[1]]$model_glm, "model_files/glm1.rds")
saveRDS(cv_results[[1]]$model_glm2, "model_files/glm2.rds")
saveRDS(cv_results[[1]]$rec_nn$steps[[3]]$key, "model_files/rec_nn_key.rds")
torch_save(cv_results[[1]]$model_nn, "model_files/nn1.pt")
torch_save(cv_results[[1]]$model_nn2, "model_files/nn2.pt")
torch_save(cv_results[[1]]$model_simple_attn, "model_files/nn_simple_attn.pt")