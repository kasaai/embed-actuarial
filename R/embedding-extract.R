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
list(
    emb_flood_zone[1:10, ], emb_flood_zone[11:20, ],
    emb_flood_zone[21:30, ], emb_flood_zone[31:40, ],
    emb_flood_zone[41:50, ]
) %>%
    Reduce(cbind, .) %>%
    knitr::kable(digits = 2)

emb_occupancy_type <- embeddings$occupancy_type %>%
    rename(embedding = e1) %>%
    filter(!level == "new")
knitr::kable(emb_occupancy_type, digits = 2)

# for multidimensional

emb_flood_zone <- embeddings$flood_zone %>%
    filter(level == "A00") %>%
    select(-level) %>%
    tidyr::pivot_longer(starts_with("e"), names_to = "dimension") %>%
    mutate(value = format(value, digits = 2))

# withr::with_options(list(knitr.kable.NA = ""), {

# })

list(
    emb_flood_zone[1:10, ], emb_flood_zone[11:20, ],
    emb_flood_zone[21:30, ],
    rbind(
        emb_flood_zone[31:34, ],
        data.frame(dimension = rep("", 6), value = rep("", 6))
    )
) %>%
    # dplyr::bind_cols() %>%
    Reduce(cbind, .) %>%
    knitr::kable(digits = 2, format = "pipe")



emb_occupancy_type <- embeddings$occupancy_type %>%
    filter(!level == "new")
knitr::kable(emb_occupancy_type, digits = 2, format = "pipe")

embeddings$number_of_floors_in_the_insured_building
embeddings %>% names()
p <- embeddings$number_of_floors_in_the_insured_building %>%
    filter(level != "new") %>% 
    ggplot() +
    geom_vline(xintercept = 0, linetype = "dashed") +
    geom_text(aes(-0.5, e1, label = round(e1, 2))) +
    geom_text(aes(1, e1, label = stringr::str_wrap(level, 40)), hjust = 0) +
    geom_point(aes(0, e1)) +
    geom_segment(aes(x = 0.8, xend = 0.1, y = e1, yend = e1),
        arrow = arrow(type = "closed", length = unit(0.075, "inches")),
        alpha = 0.5
    ) +
    coord_cartesian(xlim = c(-3, 8)) +
        coord_flip() +
        theme_void()
p2 <- embeddings$number_of_floors_in_the_insured_building %>%
    filter(level != "new") %>%
    ggplot() +
    coord_cartesian(ylim = c(-1, 1), xlim = c(NA, -1)) +
    # coord_cartesian(ylim = c(-3, 3), xlim = c(-5, 3)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_text(aes(e1, 0.1, label = round(e1, 2), angle = 45)) +
    geom_text(aes(e1, -0.1, label = stringr::str_wrap(level, 40)), hjust = 0, angle = -45) +
    geom_point(aes(e1, 0)) +
    theme_void()
    # geom_segment(aes(x = 0.8, xend = 0.1, y = e1, yend = e1),
    #     arrow = arrow(type = "closed", length = unit(0.075, "inches")),
    #     alpha = 0.5
    # ) +
    # coord_cartesian(xlim = c(-3, 8)) +
    #     theme_void()
p2
ggsave("manuscript/images/num_floors.png", plot = p2, width = 8, height = 5)

library(ggrepel)
p3 <- embeddings$flood_zone %>%
    filter(level != "new") %>%
    mutate(prefix = substr(level, 1, 1)) %>%
    ggplot(aes(e1, prefix, color = prefix, label = level)) +
    geom_point() +
    geom_label_repel() +
    theme_bw()

ggsave("manuscript/images/flood_zone.png", plot = p3, width = 8, height = 8)


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
    mutate(prefix = substr(lbls, 1, 1)) %>%
    ggplot(aes(x = PC1, y = PC2, color = prefix)) +
    geom_point() +
    geom_label_repel(aes(label = class)) +
    coord_cartesian(xlim = c(-2, 2), ylim = c(-2, 2)) +
    theme(aspect.ratio = 1) +
    theme_classic()