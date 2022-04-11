library(broom.helpers)
library(tidyverse)
library(torch)
library(ggrepel)
library(Rtsne)
source("R/table-charts-utils.R")

## Model1 relativity table

glm1 <- cv_results[[2]]$glm_simple

coefs_glm1 <- tidy_and_attach(glm1, exponentiate = TRUE) %>%
    tidy_add_reference_rows() %>%
    tidy_add_term_labels() %>%
    tidy_remove_intercept() %>%
    arrange(var_type) %>%
    filter(variable %in% c("occupancy_type", "flood_zone")) %>%
    select(variable, label, estimate) %>%
    mutate(estimate = ifelse(is.na(estimate), 1, estimate)) %>%
    rename(relativity = estimate)

coefs_glm1 %>%
    knitr::kable(digits = 2, format = "pipe")


## Model4 relativity table

glm2 <- cv_results[[2]]$model_glm_transfer

tidy_and_attach(glm2, exponentiate = FALSE) %>%
    select(term, estimate) %>%
    knitr::kable(digits = 3, format = "pipe")


## Model2 learned embeddings tables

nn1 <-  cv_results[[2]]$model_nn
analysis_data <- analysis(cvfolds$splits[[2]])
nn1_embeddings = get_embed_map(analysis_data, nn1, 6)

emb_flood_zone <- nn1_embeddings$flood_zone
emb_flood_zone$flood_zoneint = NULL
emb_flood_zone %>% setnames(names(emb_flood_zone), c("level", "embedding"))
list(
    emb_flood_zone[1:10, ], emb_flood_zone[11:20, ],
    emb_flood_zone[21:30, ], emb_flood_zone[31:40, ],
    emb_flood_zone[41:50, ]
) %>%
    Reduce(cbind, .) %>%
    knitr::kable(digits = 2, format = "pipe")

emb_occupancy_type <- nn1_embeddings$occupancy_type
emb_occupancy_type$occupancy_typeint  = NULL
emb_occupancy_type %>% setnames(names(emb_occupancy_type), c("level", "embedding"))
knitr::kable(emb_occupancy_type, digits = 2, format = "pipe")

nn1_embeddings$flood_zone = emb_flood_zone %>% copy
nn1_embeddings$occupancy_type = emb_occupancy_type %>% copy

## Model4 learned embeddings tables

nn2 <-  cv_results[[2]]$model_nn2
analysis_data <- analysis(cvfolds$splits[[2]])
nn2_embeddings = get_embed_map(analysis_data, nn2, 6)

emb_flood_zone <- nn2_embeddings$flood_zone
emb_flood_zone$flood_zoneint = NULL
emb_flood_zone %>% setnames(names(emb_flood_zone), c("level", paste0("e", 1:34)))

emb_flood_zone = emb_flood_zone%>%
    filter(level == "A00") %>%
    select(-level) %>%
    tidyr::pivot_longer(starts_with("e"), names_to = "dimension") %>%
    mutate(value = format(value, digits = 2))

list(
    emb_flood_zone[1:10, ], emb_flood_zone[11:20, ],
    emb_flood_zone[21:30, ],
    rbind(
        emb_flood_zone[31:34, ],
        data.frame(dimension = rep("", 6), value = rep("", 6))
    )
) %>%
    Reduce(cbind, .) %>%
    knitr::kable(digits = 2, format = "pipe")

emb_occupancy_type <- nn2_embeddings$occupancy_type
emb_occupancy_type$occupancy_typeint  = NULL
emb_occupancy_type %>% setnames(names(emb_occupancy_type),  c("level", paste0("e", 1:3)))

knitr::kable(emb_occupancy_type, digits = 2, format = "pipe")

nn2_embeddings$flood_zone = emb_flood_zone %>% copy
nn2_embeddings$occupancy_type = emb_occupancy_type %>% copy

## Model2 embedding plots

nn1_embeddings$occupancy_type[, e1 := embedding]
nn1_embeddings$flood_zone[, e1 := embedding]

p <- nn1_embeddings$occupancy_type %>%
    filter(level != "new") %>%
    ggplot() +
    coord_cartesian(ylim = c(-1, 0.75), xlim = c(-0.2, -0.01)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    # coord_cartesian(ylim = c(-3, 3), xlim = c(-5, 3)) +
    geom_text(aes(e1, 0.1, label = round(e1, 2), angle = 90)) +
    geom_text(aes(e1, -0.1, label = stringr::str_wrap(level, 40)), hjust = 0, angle = -90) +
    geom_point(aes(e1, 0)) +
    theme_void()
p
ggsave("manuscript/images/occupancy_type_embed.png", plot = p, width = 8, height = 5)

p <- nn1_embeddings$flood_zone %>%
    filter(level != "new") %>%
    mutate(prefix = substr(level, 1, 1)) %>%
    ggplot(aes(e1, prefix, color = prefix, label = level)) +
    geom_point() +
    # geom_label_repel() +
    theme_bw()
p
ggsave("manuscript/images/flood_zone_embed.png", plot = p, width = 8, height = 6)


## Model4 PCA and t-SNE

lbls <- nn2_embeddings$flood_zone$level
wts <- nn2_embeddings$flood_zone %>%
    select(starts_with("e")) %>%
    as.matrix()

pca <- prcomp(wts, center = TRUE, scale. = TRUE, rank. = 2)
summary(pca)

pca_mapped <- pca$x[, c("PC1", "PC2")] %>%
    as.data.frame()

p <- pca_mapped %>%
    as.data.frame() %>%
    mutate(class = lbls) %>%
    mutate(prefix = substr(lbls, 1, 1)) %>%
    filter(class != "new") %>%
    ggplot(aes(x = PC1, y = PC2, color = prefix)) +
    geom_point() +
    # geom_label_repel(aes(label = class)) +
    theme_classic()
p
ggsave("manuscript/images/flood_zone_pca.png", plot = p)

tsne_out <- Rtsne::Rtsne(wts, pca = FALSE, perplexity = 0.5, eta = 100, theta = 0, max_iter = 10000)
p_1 <- tsne_out$Y %>%
    as.data.frame() %>%
    mutate(class = lbls) %>%
    mutate(prefix = substr(lbls, 1, 1)) %>%
    filter(class != "new") %>%
    ggplot(aes(x = V1, y = V2, color = prefix)) +
    geom_point() +
    # geom_label_repel(aes(label = class)) +
    theme_classic()
p
p1 <- make_tsne_plot(wts, lbls, 2)
p2 <- make_tsne_plot(wts, lbls, 3)
p3 <- make_tsne_plot(wts, lbls, 5)
p4 <- make_tsne_plot(wts, lbls, 10)
library(patchwork)
p_all <- (p1 + theme(legend.position = "none") + p2) / (p3 + theme(legend.position = "none") + p4) + plot_layout(guides = "collect")
p_all
ggsave("manuscript/images/flood_zone_tsne.png", plot = p_all)
