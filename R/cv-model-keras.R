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

### for finetuning use first 100k
for_cv = small_data[1:100000,]
### for test use second 100k
#for_test = small_data[100000:200000,]

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

### first run - Section 4 models

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

    simple_nn = get_keras_model(hidden_layer_units = 128,hidden_layer_activations = "relu", embed = 1, for_card, train_dl, test_dl, output_list)
    preds_nn <- simple_nn$test_preds

    # Model04

    simple_nn_multidim = get_keras_model(hidden_layer_units = 128,hidden_layer_activations = "relu", embed = "multi" ,for_card, train_dl, test_dl, output_list)
    preds_nn2 <- simple_nn_multidim$test_preds

    # formula used for both tabnet and glm
    form <- amount_paid_on_building_claim ~ total_building_insurance_coverage +
      basement_enclosure_crawlspace_type + number_of_floors_in_the_insured_building +
      occupancy_type + flood_zone + primary_residence + community_rating_system_discount

    # # tabnet
    # rec_tabnet <- recipe(amount_paid_on_building_claim ~ .,
    #                      data = analysis_data %>%
    #                        select(-loss_proportion, -reported_zip_code)
    # ) %>%
    #   step_normalize(all_numeric(), -all_outcomes()) %>%
    #   step_novel(all_nominal()) %>%
    #   prep(strings_as_factors = TRUE)
    #
    # model_tabnet <- tabnet_fit(form,
    #                            data = juice(rec_tabnet), epochs = 10,
    #                            num_independent = 1, num_shared = 1, num_steps = 3,
    #                            decision_width = 1, attention_width = 1,
    #                            learn_rate = 1, verbose = TRUE, batch_size = 2000,
    #                            valid_split = 0.2
    # )
    #
    # preds_tabnet <- predict(model_tabnet, bake(rec_tabnet, assessment_data))$.pred

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

    model_glm <- glm(amount_paid_on_building_claim~.,
                     family = Gamma(link = log), data = glm_transfer_train,
                     control = list(maxit = 500))

    preds_glm_train <- predict(model_glm, glm_transfer_train, type = "response")
    preds_glm_test <- predict(model_glm, glm_transfer_test, type = "response")

    preds_glm2 <- preds_glm_test

    # analysis_data = analysis_data %>% data.table()
    # embed_map = get_embed_map(analysis_data, simple_nn$model, 6)
    # #
    # # temp = embed_map[["flood_zone"]]
    # # temp[, flood_zone_long := flood_zone]
    # # temp[, flood_zone := str_sub(flood_zone_long,1,1)]
    # # temp =     temp[, .(flood_zone_embed_1 = mean(flood_zone_embed_1)), keyby = flood_zone]
    # # embed_map[["flood_zone"]] =temp %>% copy
    #
    # train_glm = juice(rec_glm) %>% data.table()
    # train_glm[, set := "train"]
    # test_glm = bake(rec_glm, assessment_data) %>% data.table()
    # test_glm[, set := "test"]
    #
    # train_glm = rbind(train_glm, test_glm)
    #
    # # train_glm[, flood_zone_long := flood_zone]
    # # train_glm[, flood_zone := str_sub(flood_zone_long,1,1)]
    # # train_glm$flood_zone_long = NULL
    #
    # train_glm_cat = train_glm[, c(categorical_cols), with=F]
    #
    # glm_cols = c( "total_building_insurance_coverage" ,
    #               "basement_enclosure_crawlspace_type" , "number_of_floors_in_the_insured_building" ,
    #               "occupancy_type" , "flood_zone" , "primary_residence" , "community_rating_system_discount")
    #
    # for (column in categorical_cols){
    #   train_glm %>% setkeyv(eval(column))
    #   train_glm_cat %>% setkeyv(eval(column))
    #   temp = embed_map[[column]]
    #   temp = temp[, c(-2)]%>% setkeyv(eval(column))
    #   train_glm_cat = train_glm_cat %>% merge(temp)
    #   train_glm_cat[, eval(column) := NULL]
    #
    # }
    #
    # train_glm = cbind(train_glm[, c("amount_paid_on_building_claim", "set"), with=F], train_glm[, c(numeric_cols), with=F],train_glm_cat)
    # test_glm = train_glm[set == "test"]
    # train_glm = train_glm[set == "train"]
    # train_glm$set = NULL
    # test_glm$set = NULL
    #
    # model_glm <- glm(amount_paid_on_building_claim~.,
    #                  family = Gamma(link = log), data = train_glm,
    #                  control = list(maxit = 100))
    #
    # preds_glm2 <- predict(model_glm, test_glm, type = "response")

#
#     simple_attn = get_keras_model_attn(hidden_layer_units = 16, embed = 2, for_card, train_dl, test_dl, output_list)
#     preds_simple_attn <- simple_attn$test_preds
#
#     # tabtrans = get_keras_model_tabtrans(hidden_layer_units = 16, embed = 0, for_card, train_dl, test_dl, output_list)
#     # preds_tabt <- tabtrans$test_preds
#     # #

    list(#model_nn = simple_nn$model,
        #model_nn2 = nn$model,
        #model_simple_attn = simple_attn$model,
        #model_tabt = tabtrans$model,
        rec_nn = rec_nn,
        actuals = actuals,
        preds_nn = preds_nn,
        preds_nn2 = preds_nn2,
        preds_glm = preds_glm,
        preds_glm2 = preds_glm2,
        preds_glm_gaussian = preds_glm_gaussian

        # preds_simple_attn = preds_simple_attn
        # preds_tabt = preds_tabt

    )
}

cv_results <- cvfolds$splits %>%
    lapply(function(x) model_analyze_assess(x, 0.001, 30, 1024))

require(data.table)
res = cv_results %>%
    map(function(x) {
        list(
            rmse_nn = rmse(x$actuals, x$preds_nn),
            rmse_nn2 = rmse(x$actuals, x$preds_nn2),
            rmse_glm = rmse(x$actuals, x$preds_glm),
            rmse_glm2 = rmse(x$actuals, x$preds_glm2),
            rmse_glm_gaussian = rmse(x$actuals, x$preds_glm_gaussian),

            mae_nn = mae(x$actuals, x$preds_nn),
            mae_nn2 = mae(x$actuals, x$preds_nn2),
            mae_glm = mae(x$actuals, x$preds_glm),
            mae_glm2 = mae(x$actuals, x$preds_glm2),
            mae_glm_gaussian = mae(x$actuals, x$preds_glm_gaussian)



            #rmse_simple_attn = rmse(x$actuals, x$preds_simple_attn),
            #rmse_tabt = rmse(x$actuals, x$preds_tabt)

            # mgd_nn = mean_gamma_deviance(x$actuals, x$preds_nn),
            #mgd_nn2 = mean_gamma_deviance(x$actuals, x$preds_nn2)
            # mgd_simple_attn = mean_gamma_deviance(x$actuals, x$preds_simple_attn),
            # mgd_tabt = mean_gamma_deviance(x$actuals, x$preds_tabt),
            # mgd_glm = mean_gamma_deviance(x$actuals, x$preds_glm),
            # mgd_glm2 = mean_gamma_deviance(x$actuals, x$preds_glm2),
            # mgd_glm_gaussian = mean_gamma_deviance(x$actuals, pmax(x$preds_glm_gaussian, 0.01))
            #mae_simpleattn = mae(x$actuals, x$preds_simple_attn)
            #mae_nn2 = mae(x$actuals, x$preds_nn2)
            #mae_tabt = mae(x$actuals, x$preds_tabt)
        )
    }) %>% rbindlist()

res[, id:= 1:5]
res[, run := "relu 128"]
res %>% fwrite("c:/r/relu128.csv")

### Second run - Section 5 models

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


  # # tabnet
  # rec_tabnet <- recipe(amount_paid_on_building_claim ~ .,
  #                      data = analysis_data %>%
  #                        select(-loss_proportion, -reported_zip_code)
  # ) %>%
  #   step_normalize(all_numeric(), -all_outcomes()) %>%
  #   step_novel(all_nominal()) %>%
  #   prep(strings_as_factors = TRUE)
  #
  # model_tabnet <- tabnet_fit(form,
  #                            data = juice(rec_tabnet), epochs = 10,
  #                            num_independent = 1, num_shared = 1, num_steps = 3,
  #                            decision_width = 1, attention_width = 1,
  #                            learn_rate = 1, verbose = TRUE, batch_size = 2000,
  #                            valid_split = 0.2
  # )
  #
  # preds_tabnet <- predict(model_tabnet, bake(rec_tabnet, assessment_data))$.pred


  simple_attn = get_keras_model_attn(hidden_layer_units = 64, embed = 0, for_card, train_dl, test_dl, output_list)
  preds_simple_attn <- simple_attn$test_preds

  tabtrans = get_keras_model_tabtrans(hidden_layer_units = 64, embed = 0, for_card, train_dl, test_dl, output_list)
  preds_tabt <- tabtrans$test_preds


  list(#model_nn = simple_nn$model,
    #model_nn2 = nn$model,
    model_simple_attn = simple_attn$model,
    #model_tabt = tabtrans$model,
    rec_nn = rec_nn,
    actuals = actuals,
    # preds_nn = preds_nn,
    # preds_nn2 = preds_nn2,
    # preds_glm = preds_glm,
    # preds_glm2 = preds_glm2,
    # preds_glm_gaussian = preds_glm_gaussian

    preds_simple_attn = preds_simple_attn,
    preds_tabt = preds_tabt

  )
}

cv_results <- cvfolds$splits %>%
  lapply(function(x) model_analyze_assess(x, 0.001, 30, 1024))

require(data.table)
res = cv_results %>%
  map(function(x) {
    list(
      rmse_sa = rmse(x$actuals, x$preds_simple_attn),
      rmse_tt = rmse(x$actuals, x$preds_tabt),

      mae_sa = mae(x$actuals, x$preds_simple_attn),
      mae_tt = mae(x$actuals, x$preds_tabt)

    )
  }) %>% rbindlist()

res[, id:= 1:5]
res[, run := "sect5"]
res %>% fwrite("c:/r/sect5_increase_doattn16b.csv")

res[, lapply(.SD, function(x) mean(x))]


##### Third run - tabnet

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
                             decision_width =1, attention_width = 1,
                             learn_rate = 1, verbose = TRUE, batch_size = 1024,
                             valid_split = 0.2
  )

  preds_tabnet <- predict(model_tabnet, bake(rec_tabnet, assessment_data))$.pred


  list(#model_nn = simple_nn$model,
    #model_nn2 = nn$model,
    #model_simple_attn = simple_attn$model,
    #model_tabt = tabtrans$model,
    rec_nn = rec_nn,
    actuals = actuals,
    preds_tabnet = preds_tabnet
    # preds_nn2 = preds_nn2,
    # preds_glm = preds_glm,
    # preds_glm2 = preds_glm2,
    # preds_glm_gaussian = preds_glm_gaussian

    #preds_simple_attn = preds_simple_attn,
    #preds_tabt = preds_tabt

  )
}

cv_results <- cvfolds$splits %>%
  lapply(function(x) model_analyze_assess(x, 0.001, 30, 1024))

require(data.table)
res = cv_results %>%
  map(function(x) {
    list(
      rmse_tabnet = rmse(x$actuals, x$preds_tabnet),
      mae_tabnet = mae(x$actuals, x$preds_tabnet)

    )
  }) %>% rbindlist()

res[, id:= 1:5]
res[, run := "tabnet"]
res %>% fwrite("c:/r/sect5_tabnet.csv")

res[, lapply(.SD, function(x) mean(x))]













res = res %>% separate(variable, into = c("metric", "mod"), sep = "_", extra = "merge") %>%
    pivot_wider(names_from = "metric") %>%
    left_join(model_names_mapping, by = "mod") %>%
    select(-mod) %>% data.table()
res[, .(RMSE = mean(rmse), MAE = mean(mae)), keyby = .(Model)] %>%
    knitr::kable(digits = 0, format.args = list(
        big.mark = ",",
        scientific = FALSE
    ), format = "pipe")

#
# dir.create("model_files")
# saveRDS(cv_results[[1]]$model_glm, "model_files/glm1.rds")
# saveRDS(cv_results[[1]]$model_glm2, "model_files/glm2.rds")
# saveRDS(cv_results[[1]]$rec_nn$steps[[3]]$key, "model_files/rec_nn_key.rds")
# torch_save(cv_results[[1]]$model_nn, "model_files/nn1.pt")
# torch_save(cv_results[[1]]$model_nn2, "model_files/nn2.pt")
# torch_save(cv_results[[1]]$model_simple_attn, "model_files/nn_simple_attn.pt")

######### load TabTransformer
library(broom.helpers)
library(tidyverse)
library(torch)
library(ggrepel)
library(Rtsne)
source("c:/r/embed-actuarial/R/table-charts-utils.R")

model = load_model_hdf5("C:/Users/user-pc/Dropbox/keras_mod_tabt.h5")

######### Load up the data

source("c:/R/data-loading.R")
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

splits = cvfolds$splits[[1]]
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

train_dl = juiced_train_data %>% get_keras_data()
test_dl = baked_test_data %>% get_keras_data()
actuals <- assessment_data %>%
    pull(amount_paid_on_building_claim)
set.seed(42)
several_pols = baked_test_data %>% sample_n(10000) %>% data.table()
several = several_pols %>% get_keras_data()
########### check embedding
acpc = small_data %>% data.table()
acpc = acpc[, mean(amount_paid_on_building_claim), keyby = .(flood_zone)]

flood = data.table(label = analysis_data$flood_zone, code = juiced_train_data$flood_zone)
flood[, prefix := str_sub(label, 1, 1)]
flood = flood[order(label)] %>% unique()

flood_embed = model$layers[[12]] %>% get_weights()
pcas = flood_embed[[1]][1:66,] %>% princomp()
flood[, paste0("PC", 1:2) := data.table(pcas$scores[1:66,1:2])]
flood %>%
    ggplot(aes(x = PC1, y = PC2, color = prefix)) +
    geom_point() +
    geom_label_repel(aes(label = label)) +
    theme_classic()

tsne_out <- Rtsne::Rtsne(flood_embed[[1]][1:66,], pca = FALSE, perplexity =2, eta = 100, theta = 0, max_iter = 10000)
tsne_out$Y %>%
    as.data.frame() %>%
    mutate(label = flood$label) %>%
    mutate(prefix = substr(flood$label, 1, 1)) %>%
    ggplot(aes(x = V1, y = V2, color = prefix)) +
    geom_point() +
  #  geom_label_repel(aes(label = label))+
    theme_classic()

p1 <- make_tsne_plot(flood_embed[[1]][1:66,], substr(flood$label, 1, 1), 2)
p2 <- make_tsne_plot(flood_embed[[1]][1:66,], substr(flood$label, 1, 1), 3)
p3 <- make_tsne_plot(flood_embed[[1]][1:66,], substr(flood$label, 1, 1), 5)
p4 <- make_tsne_plot(flood_embed[[1]][1:66,], substr(flood$label, 1, 1), 10)
library(patchwork)
p_all <- (p1 + theme(legend.position = "none") + p2) / (p3 + theme(legend.position = "none") + p4) + plot_layout(guides = "collect")
p_all

########

model_attended = keras_model(inputs = model$inputs,
                             outputs = c(model$layers[[16]]$output, model$layers[[33]]$output,
                                         model$layers[[23]]$output,model$layers[[24]]$output))

preds = model_attended %>% predict(several)

flood = data.table(label = analysis_data$flood_zone, code = juiced_train_data$flood_zone)
flood[, prefix := str_sub(label, 1, 1)]
flood = flood[order(label)] %>% unique()
flood[,  flood_zone := code]
flood = flood[, c(1, 3, 4)] %>% setkey(flood_zone)

basement_enclosure_crawlspace_type = data.table(basement_enclosure_crawlspace = analysis_data$basement_enclosure_crawlspace_type,
                                                code = juiced_train_data$basement_enclosure_crawlspace_type)
basement_enclosure_crawlspace_type = basement_enclosure_crawlspace_type[order(basement_enclosure_crawlspace)] %>% unique()
basement_enclosure_crawlspace_type[,  basement_enclosure_crawlspace_type := code]
basement_enclosure_crawlspace_type = basement_enclosure_crawlspace_type[, c(1, 3)] %>% setkey(basement_enclosure_crawlspace_type)

flood_embed = preds[[1]][,5,]
flood_embed_context = preds[[2]][,5,]

flood_embed_pca = (preds[[1]][,5,] %>% princomp())$scores[,1:2]
flood_embed_context_pca = (preds[[2]][,5,] %>% princomp())$scores[,1:2]

several_pols[, paste0("flood_embed_score_",1:20) := data.table(flood_embed)]
several_pols[,  paste0("flood_embed_context_score_",1:20) := data.table(flood_embed_context)]
several_pols[, paste0("flood_embed_pca_", 1:2) := data.table(flood_embed_pca)]
several_pols[, paste0("flood_embed_context_pca_", 1:2) := data.table(flood_embed_context_pca)]
several_pols %>% setkey(flood_zone)
several_pols = several_pols %>% merge(flood)
several_pols %>% setkey(basement_enclosure_crawlspace_type)
several_pols = several_pols %>% merge(basement_enclosure_crawlspace_type)


several_pols %>% ggplot(aes(x = flood_embed_pca_1, y = flood_embed_context_pca_1)) +
    geom_point()+
    theme_classic()+xlab("Embedding PCA Component 1")+ylab("Contextual Embedding PCA Component 1")

ggsave("C:/R/embed-actuarial/manuscript/images/emb_vs_cembd.pdf")


several_pols %>% ggplot(aes(x = flood_embed_pca_1, y = flood_embed_context_pca_1)) +
    geom_point(aes(colour = basement_enclosure_crawlspace))+
    theme_classic()+xlab("Embedding PCA Component 1")+ylab("Contextual Embedding PCA Component 1")


ggsave("C:/R/embed-actuarial/manuscript/images/emb_vs_cembd_exp.pdf")

embed_dat = several_pols[, c(paste0("flood_embed_score_",1:20), "amount_paid_on_building_claim")]
cembed_dat = several_pols[, c(paste0("flood_embed_context_score_",1:20), "amount_paid_on_building_claim")]

fit1 = glm((amount_paid_on_building_claim)~ .,
           data = embed_dat, family = Gamma(link = log))
fit2 = glm((amount_paid_on_building_claim)~ .,
           data = cembed_dat, family = Gamma(link = log))

# fit1 %>% plot
# fit2 %>% plot
embed_dat[, pred := fit1 %>% predict(type = "response")]
cembed_dat[, pred := fit2 %>% predict(type = "response")]





data.table(`Embedding Type` = c("Embedding", "Contextual embedding"), RMSE =
               c(rmse(embed_dat$amount_paid_on_building_claim, embed_dat$pred),
                 rmse(cembed_dat$amount_paid_on_building_claim, cembed_dat$pred)),

                 MAE = c(mae(embed_dat$amount_paid_on_building_claim, embed_dat$pred),
                   mae(cembed_dat$amount_paid_on_building_claim, cembed_dat$pred))) %>%
    knitr::kable( digits = 2, format = "pipe")


######## EDA


######### Load up the data
source("c:/R/data-loading.R")
categorical_cols <- c(
    "primary_residence", "basement_enclosure_crawlspace_type",
    "number_of_floors_in_the_insured_building",
    "occupancy_type", "flood_zone"
)

numeric_cols <- c("total_building_insurance_coverage", "community_rating_system_discount")
response_col <- "amount_paid_on_building_claim"

small_data = small_data %>% data.table()

### claim size distr
### predictor variables - ggpairs
### correlation b-n claim_size and key variables
require(scales)

small_data %>% ggplot(aes(x = log(amount_paid_on_building_claim))) + geom_histogram()+
    scale_y_continuous(label=comma) + xlab("Log Amount paid on building claim") + ylab("Count")+
    theme_classic()

ggsave("C:/R/embed-actuarial/manuscript/images/claim_size.pdf")

small_data = small_data[order(amount_paid_on_building_claim)]
small_data[, perc := 1:.N/.N]


small_data %>% sample_frac(0.1) %>%  ggplot(aes(x = (amount_paid_on_building_claim), y = (1-perc))) + geom_point()+
    scale_y_log10() + scale_x_log10()+
    ggtitle("Log-log plot")+theme_classic()+ xlab("Log Amount paid on building claim") + ylab("Log of Empirical Survival Function")+
    theme_classic()

ggsave("C:/R/embed-actuarial/manuscript/images/loglog.pdf")


cats = small_data[,..categorical_cols]
cats[, id:=1:.N]
cats = cats %>% melt.data.table(id.vars = "id")
all_cats = cats[, .N, keyby = .(variable, value)]

all_plots = list()
i = 0
for (cat_var in categorical_cols){
    i = i + 1
    all_plots[[i]] = all_cats[variable == cat_var] %>%
    ggplot(aes(x = value, y = N)) + geom_col() +
    theme_classic()+
        ggtitle(cat_var) + ylab("Count") + xlab("Level")+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size = 10))+
        scale_y_continuous(label=comma)

    ggsave(paste0("C:/R/embed-actuarial/manuscript/images/", cat_var, ".pdf"), width = 10, height = 10)

}

nums = small_data[,..numeric_cols]
nums[, id:=1:.N]
nums = nums %>% melt.data.table(id.vars = "id")

all_plots = list()
i = 0
for (num_var in numeric_cols){
    i = i + 1
    all_plots[[i]] = nums[variable == num_var] %>%
        ggplot(aes(x = value)) + geom_histogram() +
        theme_classic()+
        ggtitle(num_var) + ylab("Count") + xlab("Value")+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size = 10))+
        scale_x_continuous(label=comma)

    ggsave(paste0("C:/R/embed-actuarial/manuscript/images/", num_var, ".pdf"), width = 10, height = 10)

}

skim_md <- function(x) skimr::skim_without_charts(x) %>% knitr::kable(format = "pipe", digits = 2, scientific = F)
small_data[,..categorical_cols] %>% skim_md()
small_data[,..numeric_cols] %>% skim_md()
