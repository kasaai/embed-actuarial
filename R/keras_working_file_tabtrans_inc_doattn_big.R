get_keras_model_tabtrans = function(hidden_layer_units = 64, embed = 0, for_card, train_dl, test_dl, output_list) {

  k_clear_session()

  input_list = list()
  embedding_list = list()
  cardinalities = list()
  count = 0

  get_embed_dim = function(cards, embed){
    if (embed == 1) {1}
    else{
      if(embed ==0) {12}
      else {(trunc(cards/2)+1)}
    }
  }

  for (column in categorical_cols){
    count = count + 1
    cardinalities[[column]] = for_card[, max(get(column))]
    input_list[[count]] = layer_input(shape = 1, dtype = "int32", name = column)
    embed_dim = get_embed_dim(cardinalities[[column]], embed)
    embedding_list[[count]] = input_list[[count]] %>%
      layer_embedding(input_dim = cardinalities[[column]]*2+1, output_dim =embed_dim,name = paste0(column, "_embed"))
  }

  input_list[[count+1]] = layer_input(shape = numeric_cols %>% length, dtype = "float32", name = "num_input")
  input_list[[count+2]] = layer_input(shape = 5, dtype = "int32", name = "col_idx")

  col_idx_int = input_list[[count+2]] %>%
    layer_reshape(c(5,1), name = "col_idx_int")

  embeddings = embedding_list %>% layer_concatenate(axis = 1)
  col_embed = layer_embedding(input_dim = 6, output_dim = 4)
  col_embeds = col_idx_int %>% time_distributed(col_embed) %>% layer_reshape(c(5,4), name = "col_embeds")
  all_embeds = list(embeddings, col_embeds) %>% layer_concatenate(axis = 2)

  attn_heads = list()

  for (i in 1:2){
    key = all_embeds %>% (layer = layer_dense(units = 16))
    query = all_embeds %>% (layer = layer_dense(units = 16))
    value = all_embeds %>% (layer = layer_dense(units = 16))
    attn = layer_attention(list(key, query, value), use_scale = T)%>%
      layer_dropout(rate = 0.5)
    attn_heads[[i]] = attn
  }

  multi_head_attn = attn_heads %>% layer_concatenate(axis=2) %>% layer_dense(units = 16)
  stage1 = list(all_embeds, multi_head_attn) %>% layer_add()%>% layer_layer_normalization()

  stage2 = stage1  %>% layer_dense(units = 16*4, activation = "linear") %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(rate = 0.5) %>% layer_dense(units = 16)

  stage3 = list(stage1, stage2) %>% layer_add() %>% layer_layer_normalization()

  attn_flat = stage3 %>% layer_flatten() %>% layer_dropout(rate = 0.05)

  main = list(attn_flat, input_list[[count+1]]) %>% layer_concatenate() %>%
    layer_dense(units = 64, activation = "tanh") %>%
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

  lr_call = callback_reduce_lr_on_plateau(monitor = "val_loss", patience = 50, verbose = 1, cooldown = 1,
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
