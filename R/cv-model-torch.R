library(tidyverse)
library(recipes)
library(rsample)
library(torch)
library(data.table)
library(keras)
require(tensorflow)
tf$compat$v1 = T

source("R/model-utils.R")
source("R/data-loading.R")
source("R/keras_working_file.R")
source("R/keras_working_file_attn.R")
source("R/keras_working_file_tabtrans.R")

categorical_cols <- c(
    "primary_residence", "basement_enclosure_crawlspace_type",
    "number_of_floors_in_the_insured_building",
    "occupancy_type", "flood_zone"
)

numeric_cols <- c("total_building_insurance_coverage", "community_rating_system_discount")
response_col <- "amount_paid_on_building_claim"

set.seed(420)
cvfolds <- small_data %>% #sample_frac(0.01) %>%
    rsample::vfold_cv(v = 5)

### t remove later
#
# splits = cvfolds$splits[[1]]
#
# learning_rate = 0.001
# epochs = 10
# batch_size = 5000

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

    #ds <- flood_dataset(juice(rec_nn), categorical_cols, numeric_cols, response_col, env)
    juiced_train_data <- juice(rec_nn)
    baked_test_data <- bake(rec_nn, assessment_data)
    for_card = juiced_train_data %>% data.table()
    for_card_test = baked_test_data %>% data.table()
    #test_ds <- flood_dataset(baked_test_data, categorical_cols, numeric_cols)

    get_keras_data = function(dat){
        dat = dat %>% data.table
        dat_list = list()
        for (column in categorical_cols) dat_list[[column]] = as.matrix(dat[, get(column)])
        dat_list[["num_input"]] = as.matrix(dat[, c(numeric_cols), with = F]) %>% unname
        dat_list[["num_input"]] = as.matrix(dat[, c(numeric_cols), with = F]) %>% unname
        dat_list[["col_idx"]] = unname(as.matrix(reshape::untable(data.table(t(1:5)),dat[,.N])))
        dat_list
    }

    get_output = function(train, test, column = "amount_paid_on_building_claim"){
        min_store = train[, min(get(column))]
        max_store = train[, max(get(column))]
        y_train = train[, (get(column) - min_store)/(max_store - min_store)]
        y_test = test[, (get(column) - min_store)/(max_store - min_store)]
        return(list(min = min_store, max = max_store, y_train = y_train, y_test = y_test))

    }

    output_list = get_output(for_card, for_card_test)

    # train_dl <- ds %>% dataloader(batch_size = batch_size, shuffle = TRUE)
    # # valid_dl <- valid_ds %>% dataloader(batch_size = batch_size, shuffle = TRUE)
    # test_dl <- test_ds %>% dataloader(batch_size = batch_size, shuffle = FALSE)

    train_dl = juiced_train_data %>% get_keras_data()
    test_dl = baked_test_data %>% get_keras_data()
    actuals <- assessment_data %>%
    pull(amount_paid_on_building_claim)


    # model_tabt <- tabtransformer(env$cardinalities, length(numeric_cols),
    #     embedding_dim = 2, num_heads = 1, fc_units = 8
    # )
    #
    # optimizer <- optim_adam(model_tabt$parameters, lr = learning_rate)
    #
    # train_loop(model_tabt, train_dl, valid_dl, epochs, optimizer)
    #
    # replace_unseen_level_weights_(model_tabt$col_embedder$embeddings)
    #
    # preds_tabt <- get_preds(model_tabt, test_dl)
    #

    # model <- simple_net(
    #     env$cardinalities, length(numeric_cols),
    #     units = 8, fn_embedding_dim = function(x) 1
    # )
    #
    # optimizer <- optim_adam(model$parameters, lr = learning_rate, weight_decay = 0)
    #
    # train_loop(model, train_dl, valid_dl, epochs, optimizer)
    #
    # replace_unseen_level_weights_(model$embedder$embeddings)

    simple_nn = get_keras_model(hidden_layer_units = 64, embed = 1)
    preds_nn <- simple_nn$test_preds

    # model2 <- simple_net(env$cardinalities, length(numeric_cols),
    #     units = 8,
    #     fn_embedding_dim = function(x) ceiling(x / 2)
    # )
    #
    # optimizer <- optim_adam(model2$parameters, lr = learning_rate, weight_decay = 0)
    #
    # train_loop(model2, train_dl, valid_dl, epochs, optimizer)
    #
    # replace_unseen_level_weights_(model2$embedder$embeddings)

    nn = get_keras_model(hidden_layer_units = 64, embed = 0)
    preds_nn2 <- nn$test_preds
#
#     model_simple_attn <- simple_net_attn(env$cardinalities, length(numeric_cols),
#                          units = 8)
#
#     optimizer <- optim_adam(model_simple_attn$parameters, lr = learning_rate, weight_decay = 0)
#
#     train_loop(model_simple_attn, train_dl, valid_dl, epochs, optimizer)
#
#     replace_unseen_level_weights_(model_simple_attn$embedder$embeddings)

    #preds_simple_attn <- get_preds(model_simple_attn, test_dl)

    simple_attn = get_keras_model_attn(hidden_layer_units = 64, embed = 0)
    preds_simple_attn <- simple_attn$test_preds

    tabtrans = get_keras_model_tabtrans(hidden_layer_units = 64, embed = 0)
    preds_tabt <- tabtrans$test_preds

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


    list(
        model_nn = simple_nn$model,
        model_nn2 = nn$model,
        model_simple_attn = simple_attn$model,
        model_tabt = tabtrans$model,
        rec_nn = rec_nn,
        model_glm = model_glm,
        rec_glm = rec_glm,
        model_glm_gaussian = model_glm_gaussian,
        model_glm2 = model_glm2,
        rec_glm2 = rec_glm2,
        actuals = actuals,
        preds_nn = preds_nn,
        preds_nn2 = preds_nn2,
        preds_simple_attn = preds_simple_attn,
        preds_tabt = preds_tabt,
        preds_glm = preds_glm,
        preds_glm_gaussian = preds_glm_gaussian,
        preds_glm2 = preds_glm2
    )
}

cv_results <- cvfolds$splits %>%
    lapply(function(x) model_analyze_assess(x, 0.5, 30, 1000))

cv_results %>%
    map(function(x) {
        list(
            rmse_nn = rmse(x$actuals, x$preds_nn),
            rmse_nn2 = rmse(x$actuals, x$preds_nn2),
            rmse_simple_attn = rmse(x$actuals, x$preds_simple_attn),
            rmse_tabt = rmse(x$actuals, x$preds_tabt),
            rmse_glm = rmse(x$actuals, x$preds_glm),
            rmse_glm2 = rmse(x$actuals, x$preds_glm2),
            rmse_glm_gaussian = rmse(x$actuals, pmax(x$preds_glm_gaussian, 0.01)),
            mgd_nn = mean_gamma_deviance(x$actuals, x$preds_nn),
            mgd_nn2 = mean_gamma_deviance(x$actuals, x$preds_nn2),
            mgd_simple_attn = mean_gamma_deviance(x$actuals, x$preds_simple_attn),
            mgd_tabt = mean_gamma_deviance(x$actuals, x$preds_tabt),
            mgd_glm = mean_gamma_deviance(x$actuals, x$preds_glm),
            mgd_glm2 = mean_gamma_deviance(x$actuals, x$preds_glm2),
            mgd_glm_gaussian = mean_gamma_deviance(x$actuals, pmax(x$preds_glm_gaussian, 0.01))
        )
    })

dir.create("model_files")
saveRDS(cv_results[[1]]$model_glm, "model_files/glm1.rds")
saveRDS(cv_results[[1]]$model_glm2, "model_files/glm2.rds")
saveRDS(cv_results[[1]]$rec_nn$steps[[3]]$key, "model_files/rec_nn_key.rds")
torch_save(cv_results[[1]]$model_nn, "model_files/nn1.pt")
torch_save(cv_results[[1]]$model_nn2, "model_files/nn2.pt")
torch_save(cv_results[[1]]$model_simple_attn, "model_files/nn_simple_attn.pt")
