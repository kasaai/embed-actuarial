extract_embeddings <- function(model, key) {
    level_integer_mappings <- key %>%
        lapply(function(x) mutate(x, integer = as.integer(integer + 1L)))
    model$embedder$embeddings %>%
        (function(x) lapply(1:length(categorical_cols), function(i) x[[i]])) %>%
        setNames(categorical_cols) %>%
        purrr::imap(function(m, v) {
            outputs <- level_integer_mappings[[v]]$integer %>%
                torch_tensor(device = "cpu") %>%
                m() %>%
                (function(x) x$to(device = "cpu")) %>%
                as.array()
            mapping_tbl <- level_integer_mappings[[v]]
            cbind(
                level = mapping_tbl$value,
                as.data.frame(outputs) %>% setNames(paste0("e", 1:ncol(outputs)))
            )
        })
}