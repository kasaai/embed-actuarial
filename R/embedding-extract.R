model <- cv_results[[1]]$model_nn
level_integer_mappings <- cv_results[[1]]$rec_nn$steps[[3]]$key %>%
    lapply(function(x) mutate(x, integer = as.integer(integer + 1L)))

embeddings <- model$embedder$embeddings %>%
    (function(x) lapply(1:length(categorical_cols), function(i) x[[i]])) %>%
    setNames(categorical_cols) %>%
    purrr::imap(function(m, v) {
        outputs <- level_integer_mappings[[v]]$integer %>%
            torch_tensor(device = "cuda") %>%
            m() %>%
            (function(x) x$to(device = "cpu")) %>% 
            as.array()
        mapping_tbl <- level_integer_mappings[[v]] 
        cbind(
            level = mapping_tbl$value,
            as.data.frame(outputs) %>% setNames(paste0("e", 1:ncol(outputs)))
        )
    })


emb_flood_zone <- embeddings$flood_zone %>%
    rename(embedding = e1)
list(emb_flood_zone[1:10, ], emb_flood_zone[11:20, ],
     emb_flood_zone[21:30, ], emb_flood_zone[31:40, ],
     emb_flood_zone[41:50, ]) %>%
    Reduce(cbind, .) %>%
    knitr::kable(digits = 2)

emb_occupancy_type <- embeddings$occupancy_type %>%
    rename(embedding = e1) %>% 
    filter(!level == "new")
knitr::kable(emb_occupancy_type, digits = 2)

embeddings$condominium_indicator %>%
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

embeddings$flood_zone %>% dim()

lbls <- embeddings$flood_zone$level
wts <- embeddings$flood_zone %>%
    select(starts_with("e")) %>%
    as.matrix()
pca <- prcomp(wts, center = TRUE, scale. = TRUE, rank = 3)$x[, c("PC1", "PC2", "PC3")] %>%
    as.data.frame()

library(ggrepel)

ggplot(data = pca, aes(x = PC1, y = PC2)) +
    geom_point()

rayshader::plot_gg(ggp)

pca %>%
    as.data.frame() %>%
    mutate(class = lbls) %>%
    mutate(prefix = substr(lbls, 1, 1))  %>%  
  ggplot(aes(x = PC1, y = PC2, color = prefix)) +
  geom_point() +
  geom_label_repel(aes(label = class)) + 
  coord_cartesian(xlim = c(-2, 2), ylim = c(-2, 2)) +
  theme(aspect.ratio = 1) +
  theme_classic()