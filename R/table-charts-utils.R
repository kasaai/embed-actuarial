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

make_tsne_plot <- function(wts, lbls, perplexity) {
    tsne_out <- Rtsne::Rtsne(wts, pca = FALSE, perplexity = perplexity, eta = 100, theta = 0, max_iter = 10000)
    p <- tsne_out$Y %>%
        as.data.frame() %>%
        mutate(class = lbls) %>%
        mutate(prefix = substr(lbls, 1, 1)) %>%
        filter(class != "new") %>%
        ggplot(aes(x = V1, y = V2, color = prefix)) +
        geom_point() +
        theme_classic()
    p
}
