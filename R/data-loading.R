claims <- cellar::cellar_pull("nfip_claims") %>%
    filter(year_of_loss > 2000)

set.seed(42069)
small_data <- claims %>%
    filter(
        amount_paid_on_building_claim > 0,
        total_building_insurance_coverage > 0
    ) %>%
    sample_n(300000) %>%
    select(
        amount_paid_on_building_claim, total_building_insurance_coverage,
        reported_zip_code, primary_residence, basement_enclosure_crawlspace_type,
        number_of_floors_in_the_insured_building, occupancy_type,
        community_rating_system_discount, flood_zone
    ) %>%
    filter(across(everything(), ~!is.na(.x))) %>%
    mutate(loss_proportion = pmin(amount_paid_on_building_claim / total_building_insurance_coverage, 1))

get_keras_data = function(dat){
  dat = dat %>% data.table
  dat_list = list()
  for (column in categorical_cols) dat_list[[column]] = as.matrix(dat[, get(column)])
  dat_list[["num_input"]] = as.matrix(dat[, c(numeric_cols), with = F]) %>% unname

  ### add column identifiers for TabTransformer

  dat_list[["col_idx"]] = unname(as.matrix(reshape::untable(data.table(t(1:5)),dat[,.N])))
  dat_list
}

get_embed_map = function(analysis_data, model, layer_start){
  dat_list = list()
  analysis_data = analysis_data %>% data.table
  i = layer_start - 1
  for (column in categorical_cols) {

    i = i + 1
    temp_lookup = data.table(cbind(analysis_data[, get(column)], analysis_data[, as.integer(as.factor(get(column)))])) %>% unique()
    temp_lookup %>% setnames(temp_lookup %>% names, c(column, paste0(column, "int")))%>% setkeyv(eval(column))
    embeds = data.table((model$layers[[i]] %>% get_weights())[[1]])
    embeds=embeds[2:((analysis_data[, as.integer(as.factor(get(column)))] %>% max())+1)]
    embeds_dim = ((model$layers[[i]] %>% get_weights())[[1]] %>% dim)[2]
    temp_lookup[,  paste0(column, "_embed", "_", 1:embeds_dim) := data.table(as.matrix(embeds))]

    dat_list[[column]] = temp_lookup %>% copy

  }

  dat_list

}

get_output = function(train, test, column = "amount_paid_on_building_claim"){
  min_store = train[, min(get(column))]
  max_store = train[, max(get(column))]
  y_train = train[, (get(column) - min_store)/(max_store - min_store)]
  y_test = test[, (get(column) - min_store)/(max_store - min_store)]
  return(list(min = min_store, max = max_store, y_train = y_train, y_test = y_test))

}
