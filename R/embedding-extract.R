model
level_integer_mappings <- rec$steps[[3]]$key %>%
    lapply(function(x) mutate(x, integer = as.integer(integer + 1L)))

embeddings <- model$embedder$embeddings %>%
    (function(x) lapply(1:length(cat_cols), function(i) x[[i]])) %>%
    setNames(cat_cols) %>%
    purrr::imap(function(m, v) {
        outputs <- level_integer_mappings[[v]]$integer %>%
            torch_tensor() %>%
            m() %>%
            as.array()
        mapping_tbl <- level_integer_mappings[[v]] 
        cbind(
            level = mapping_tbl$value,
            as.data.frame(outputs) %>% setNames(paste0("e", 1:ncol(outputs)))
        )
    })

embeddings$flood_zone %>%
  ggplot() +
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_text(aes(-0.5, e1, label = round(e1, 2))) +
  geom_text(aes(1, e1, label = level), hjust = 0) +
  geom_point(aes(0, e1)) +
  geom_segment(aes(x = 0.8, xend = 0.1, y = e1, yend = e1),
               arrow = arrow(type = "closed", length = unit(0.075, "inches")),
               alpha = 0.5) +
  coord_cartesian(xlim = c(-3, 5)) +
  theme_void()

embeddings$flood_zone %>%
  mutate(prefix = substr(level, 1, 1)) %>%
  ggplot() +
  geom_point(aes(e1, e2, color = prefix), alpha = 1)

embeddings$state %>%
    ggplot() +
    geom_point(aes(e1, e2)) +
    geom_text(aes(e1, e2, label = level))
