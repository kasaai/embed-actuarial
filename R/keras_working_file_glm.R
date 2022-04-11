get_glm_embed = function(model, for_card, train_dl, test_dl, output_list) {

 k_clear_session()

 model <- keras_model(
  inputs = model$inputs,
  outputs = model$layers[[18]]$output)

  adam = optimizer_adam(lr=0.001)

  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = adam
  )

  concat_nn_train = model %>% predict(train_dl) %>% data.table()
  glm_train = as.matrix(output_list$y_train)*(output_list$max - output_list$min)+output_list$min
  concat_nn_train$amount_paid_on_building_claim = glm_train

  concat_nn_test = model %>% predict(test_dl) %>% data.table()

  model_glm <- glm(amount_paid_on_building_claim~.,
                   family = Gamma(link = log), data = concat_nn_train,
                   control = list(maxit = 500))

  preds_glm_train <- predict(model_glm, concat_nn_train, type = "response")
  preds_glm_test <- predict(model_glm, concat_nn_test, type = "response")

  results = list(model = model, train_preds = preds_glm_train, test_preds = preds_glm_test, glm_train=concat_nn_train %>% copy, glm_test=concat_nn_test %>% copy)

  results

}
