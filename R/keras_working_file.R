get_keras_model = function(hidden_layer_units = 64, hidden_layer_activations = "relu",
                           embed = 1, for_card, train_dl, test_dl, output_list) {

k_clear_session()

input_list = list()
embedding_list = list()
cardinalities = list()
count = 0

get_embed_dim = function(cards, embed){
  if (embed == 1) {1}
  else{
  (trunc(cards/2)+1)
  }
}

for (column in categorical_cols){
  count = count + 1
  cardinalities[[column]] = for_card[, max(get(column))]
  input_list[[count]] = layer_input(shape = 1, dtype = "int32", name = column)
  embed_dim = get_embed_dim(cardinalities[[column]], embed)
  embedding_list[[count]] = input_list[[count]] %>%
    layer_embedding(input_dim = cardinalities[[column]]*2+1, output_dim =embed_dim)  %>%
    layer_flatten(name = paste0(column, "_embed"))
}

input_list[[count+1]] = layer_input(shape = numeric_cols %>% length, dtype = "float32", name = "num_input")

embeddings = embedding_list %>% layer_concatenate()
main = list(embeddings, input_list[[count+1]]) %>% layer_concatenate() %>%
  layer_dense(units = hidden_layer_units, activation = hidden_layer_activations) %>%
  layer_dropout(rate = 0.025)

output = main %>%
  layer_dense(units = 1, activation = "sigmoid", name = "output")

model <- keras_model(
  inputs = input_list,
  outputs = c(output))

adam = optimizer_adam(lr=0.001)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = adam
)

lr_call = callback_reduce_lr_on_plateau(monitor = "val_loss", patience = 50, verbose = 1, cooldown = 5,
                                        factor = 0.9, min_lr = 0.0001)
mod_save = callback_model_checkpoint(filepath = paste0("c:/r/keras_mod.h5"), verbose = T, save_best_only = T)

fit <- model %>% fit(x = train_dl,
                     y= list(output = as.matrix(output_list$y_train)),
                     epochs=150,
                     batch_size=1024,
                     verbose=2,
                     validation_split = 0.05,
                     shuffle=T,
                     callbacks = list(lr_call, mod_save))

model = load_model_hdf5("c:/r/keras_mod.h5")

preds_nn_train = model %>% predict(train_dl)
preds_nn_train = preds_nn_train*(output_list$max - output_list$min)+output_list$min

preds_nn = model %>% predict(test_dl)
preds_nn = preds_nn*(output_list$max - output_list$min)+output_list$min

results = list(model = model, train_preds = preds_nn_train, test_preds = preds_nn)

}

gamma_loss = function(y, y_hat) {
  y = y +  k_epsilon()
  y_hat = y_hat +  k_epsilon()
  temp = (y-y_hat)/y_hat - k_log(y/y_hat + k_epsilon())
  return(k_mean(temp))

}
