flood_dataset <- dataset(
    "flood",
    initialize = function(df, categorical_cols, numeric_cols, response_col = NULL, env = NULL) {
        device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"
        self$is_train <- if (!is.null(response_col)) TRUE else FALSE
        self$xcat <- df[categorical_cols] %>%
            as.matrix() %>%
            `+`(1L) %>%
            torch_tensor(device = device)
        if (!is.null(env)) {
            assign("cardinalities", sapply(df[categorical_cols], function(x) max(x) + 2), envir = env)
        }
        self$xnum <- df[numeric_cols] %>%
            as.matrix() %>%
            torch_tensor(device = device)

        if (self$is_train) {
            self$y <- df[[response_col]] %>%
                as.matrix() %>%
                torch_tensor(device = device)
        }

        self
    },
    .getitem = function(i) {
        xcat <- self$xcat[i, ]
        xnum <- self$xnum[i, ]

        if (self$is_train) {
            y <- self$y[i]

            list(x = list(xcat, xnum), y = y)
        } else {
            list(x = list(xcat, xnum))
        }
    },
    .length = function() {
        nrow(self$xnum)
    }
)

embedding_module <- nn_module(
    initialize = function(cardinalities, fn_embedding_dim, max_norm = NULL, norm_type = 2) {
        self$embeddings <- nn_module_list(lapply(
            cardinalities,
            function(x) {
                nn_embedding(
                    num_embeddings = x, embedding_dim = fn_embedding_dim(x),
                    max_norm = max_norm, norm_type = norm_type
                )
            }
        ))
    },
    forward = function(x) {
        embedded <- vector(mode = "list", length = length(self$embeddings))
        for (i in seq_along(self$embeddings)) {
            embedded[[i]] <- self$embeddings[[i]](x[, i])
        }
        torch_cat(embedded, dim = 2)
    }
)

simple_net <- nn_module(
    "simple_net",
    initialize = function(cardinalities,
                          num_numerical,
                          units = 16,
                          fn_embedding_dim = function(x) ceiling(x / 2)) {
        self$embedder <- embedding_module(cardinalities, fn_embedding_dim)
        sum_embedding_dim <- sapply(cardinalities, fn_embedding_dim) %>%
            sum()
        self$fc <- nn_linear(sum_embedding_dim + num_numerical, units)
        self$output <- nn_linear(units, 1)
        device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"
        self$to(device = device)
    },
    forward = function(xcat, xnum) {
        embedded <- self$embedder(xcat)
        all <- torch_cat(list(embedded, xnum$to(dtype = torch_float())), dim = 2)
        all %>%
            self$fc() %>%
            nnf_relu() %>%
            self$output() %>%
            nnf_softplus()
    }
)

train_loop_alternate <- function(model, train_dl, valid_dl, epochs, optimizer) {
    for (epoch in seq_len(epochs)) {
        model$train()
        train_losses <- c()

        odd_epoch <- epoch %% 2 == 1

        for (p in model$embedder$parameters) {
            p$requires_grad_(odd_epoch)
        }

        for (p in model$output$parameters) {
            p$requires_grad_(!odd_epoch)
        }

        for (b in enumerate(train_dl)) {
            optimizer$zero_grad()
            output <- model(b$x[[1]], b$x[[2]])
            loss <- nnf_mse_loss(output, b$y)
            loss$backward()

            # if (odd_epoch) {
            #     nn_utils_clip_grad_value_(model$embedder$parameters, 1)
            # } else {
            #     nn_utils_clip_grad_value_(model$output$parameters, 1)
            # }

            optimizer$step()
            train_losses <- c(train_losses, loss$item())
        }

        model$eval()
        valid_losses <- c()

        for (b in enumerate(valid_dl)) {
            output <- model(b$x[[1]], b$x[[2]])
            loss <- nnf_mse_loss(output, b$y)
            valid_losses <- c(valid_losses, loss$item())
        }

        cat(sprintf(
            "Loss at epoch %d: training: %3f, validation: %3f\n", epoch,
            mean(train_losses), mean(valid_losses)
        ))
    }
}

train_loop <- function(model, train_dl, valid_dl, epochs, optimizer) {
    device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"
    for (epoch in seq_len(epochs)) {
        model$train()
        train_losses <- c()

        for (b in enumerate(train_dl)) {
            optimizer$zero_grad()
            output <- model(b$x[[1]]$to(device = device), b$x[[2]]$to(device = device))
            loss <- nnf_mse_loss(output, b$y$to(device = device))
            loss$backward()
            optimizer$step()
            train_losses <- c(train_losses, loss$item())
        }

        model$eval()
        valid_losses <- c()

        for (b in enumerate(valid_dl)) {
            output <- model(b$x[[1]]$to(device = device), b$x[[2]]$to(device = device))
            loss <- nnf_mse_loss(output, b$y$to(device = device))
            valid_losses <- c(valid_losses, loss$item())
        }

        cat(sprintf(
            "Loss at epoch %d: training: %3f, validation: %3f\n", epoch,
            mean(train_losses), mean(valid_losses)
        ))
    }
}

get_preds <- function(model, dl) {
    device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"
    preds <- numeric(0)
    for (b in enumerate(dl)) {
        preds <- c(
            preds,
            model(b$x[[1]]$to(device = device), b$x[[2]]$to(device = device))$to(device = "cpu") %>% as.array()
        )
    }
    preds
}

replace_unseen_level_weights_ <- function(embeddings) {
    for (emb in as.list(embeddings)) {
        emb_num_lvls <- dim(emb$weight)[[1]]
        emb_dim <- dim(emb$weight)[[2]]
        for (j in seq_len(emb_dim)) {
            median_wt <- torch_median(emb$weight[1:(emb_num_lvls - 1), j])
            emb$weight[emb_num_lvls, j] <- median_wt
        }
    }

    embeddings
}

map_cats_to_embeddings <- function(data, keys) {
    for (v in names(keys)) {
        mapping_table <- keys[[v]] %>%
            select(value, embedding) %>%
            rename(!!v := value)
        data <- data %>%
            left_join(mapping_table, by = v) %>%
            select(-!!v) %>%
            rename(!!v := embedding)
    }

    data
}

key_with_embeddings <- function(embeddings, key) {
    device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"
    Map(
        function(embedder, key) {
            # key$integer
            embedding <- key$integer %>%
                `+`(1L) %>%
                torch_tensor(dtype = torch_int(), device = device) %>%
                embedder() %>%
                (function(x) x$to(device = "cpu")) %>% 
                as.numeric()
            key[["embedding"]] <- embedding
            key
        },
        as.list(embeddings),
        key
    ) %>%
        setNames(names(key))
}