flood_dataset <- dataset(
    "flood",
    initialize = function(df, categorical_cols, numeric_cols, coverage_col, response_col = NULL, env = NULL) {
        # device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"
        device <- "cpu"
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

        self$xcoverage <- df[coverage_col] %>%
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
        xcat <- self$xcat[i, , drop = TRUE]
        xnum <- self$xnum[i, , drop = TRUE]
        xcoverage <- self$xcoverage[i]

        if (self$is_train) {
            y <- self$y[i]

            list(x = list(xcat, xnum, xcoverage), y = y)
        } else {
            list(x = list(xcat, xnum, xcoverage))
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
        in_units <- sum_embedding_dim + num_numerical
        self$fc <- nn_linear(in_units, units)
        self$output <- nn_linear(units, 1)
        self
    },
    forward = function(xcat, xnum, xcoverage) {
        embedded <- self$embedder(xcat)
        all <- torch_cat(list(embedded, xnum$to(dtype = torch_float())), dim = 2)
        out <- all %>%
            self$fc() %>%
            nnf_relu()

        ratio <- out %>%
            self$output() %>%
            nnf_sigmoid()

        ratio * xcoverage
    }
)


simple_net_attn <- nn_module(
    "simple_net_attn",
    initialize = function(cardinalities,
                          num_numerical,
                          units = 16,
                          embed_dim = 10,
                          fn_embedding_dim = function(x) embed_dim) {
        self$embedder <- embedding_module(cardinalities, fn_embedding_dim)
        sum_embedding_dim <- sapply(cardinalities, fn_embedding_dim) %>%
            sum()
        self$embed_dim <- fn_embedding_dim()
        self$attn <- nn_multihead_attention(embed_dim = embed_dim, num_heads = 1, dropout = 0.02)
        self$fc <- nn_linear(sum_embedding_dim + num_numerical, units)
        self$output <- nn_linear(units, 1)
        self
    },
    forward = function(xcat, xnum, xcoverage) {
        embedded <- self$embedder(xcat)
        shapes <- embedded$shape
        embedded_reshape <- embedded$view(list(model$embed_dim, embedded$shape[1], model$embed_dim))
        embedded_attention <- model$attn(embedded_reshape, embedded_reshape, embedded_reshape)
        embedded_attended <- embedded_attention[[1]]
        embedded_attended <- embedded_attended$view(list(embedded$shape[1], model$embed_dim * model$embed_dim))
        all <- torch_cat(list(embedded_attended, xnum$to(dtype = torch_float())), dim = 2)
        ratio <- all %>%
            self$fc() %>%
            nnf_relu() %>%
            self$output() %>%
            nnf_sigmoid()

        ratio * xcoverage
    }
)

train_loop <- function(model, train_dl, valid_dl = NULL, epochs, optimizer, patience = 2, model_name) {
    # print(valid_dl)
    # device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"
    device <- "cpu"
    best_loss <- NULL
    counter <- 0
    for (epoch in seq_len(epochs)) {
        model$train()
        train_losses <- c()

        coro::loop(for (b in train_dl) {
            optimizer$zero_grad()
            output <- model(b$x[[1]]$to(device = device), b$x[[2]]$to(device = device), b$x[[3]]$to(device = device))
            loss <- nnf_mse_loss(output, b$y$to(device = device))
            loss$backward()
            optimizer$step()
            train_losses <- c(train_losses, loss$item())
        })

        model$eval()
        valid_losses <- c()

        coro::loop(for (b in valid_dl) {
            output <- model(b$x[[1]]$to(device = device), b$x[[2]]$to(device = device), b$x[[3]]$to(device = device))
            loss <- nnf_mse_loss(output, b$y$to(device = device))
            valid_losses <- c(valid_losses, loss$item())
        })

        cat(sprintf(
            "Loss at epoch %d: training: %3f, validation: %3f\n", epoch,
            mean(train_losses), mean(valid_losses)
        ))

        if (is.null(best_loss)) {
            best_loss <- mean(valid_losses)
            torch_save(model, paste0("model_files/", model_name, ".pt"))
            cat("**Saved model\n")
        } else {
            if (mean(valid_losses) < best_loss) {
                best_loss <- mean(valid_losses)
                torch_save(model, paste0("model_files/", model_name, ".pt"))
                cat("**Saved model\n")
                counter <- 0
            } else {
                counter <- counter + 1
            }
        }
        if (counter >= patience) {
            cat("**Early stopping!\n")
            break
        }
    }
}

get_preds <- function(model, dl) {
    model$eval()
    # device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"
    device <- "cpu"
    preds <- numeric(0)
    for (b in enumerate(dl)) {
        preds <- c(
            preds,
            model(
                b$x[[1]]$to(device = device),
                b$x[[2]]$to(device = device),
                b$x[[3]]$to(device = device)
            )$to(device = "cpu") %>% as.array()
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
    # device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"
    device <- "cpu"
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

embedding_with_position <- nn_module(
    initialize = function(cardinalities, embedding_dim = 2, max_norm = NULL, norm_type = 2) {
        self$embeddings <- nn_module_list(lapply(
            cardinalities,
            function(x) {
                nn_embedding(
                    num_embeddings = x, embedding_dim = embedding_dim,
                    max_norm = max_norm, norm_type = norm_type
                )
            }
        ))

        self$position_embeddings <- nn_parameter(
            torch_zeros(c(1, length(cardinalities), 1)),
            requires_grad = TRUE
        )
        self
    },
    forward = function(x) {
        device <- "cpu"
        embedded <- vector(mode = "list", length = length(self$embeddings))
        for (i in seq_along(self$embeddings)) {
            embedded[[i]] <- self$embeddings[[i]](x[, i])
        }
        torch_cat(list(
            torch_stack(embedded, dim = 2),
            torch_repeat_interleave(self$position_embeddings,
                torch_tensor(x$size()[[1]], dtype = torch_long())$to(device = device),
                dim = 1
            )
        ), dim = 3)
    }
)

mlp <- nn_module(
    "mlp",
    initialize = function(in_dim, fc_units) {
        self$linear1 <- nn_linear(in_dim, fc_units)
        self$linear2 <- nn_linear(fc_units, 1)
    },
    forward = function(x) {
        x %>%
            self$linear1() %>%
            nnf_relu() %>%
            self$linear2()
    }
)

tabtransformer <- nn_module(
    "tabtransformer",
    initialize = function(cardinalities, num_numerical, embedding_dim = 10, num_heads = 3, fc_units = 32) {
        self$col_embedder <- embedding_with_position(cardinalities, embedding_dim)
        self$attn <- nn_multihead_attention(embedding_dim + 1, num_heads, dropout = 0.02)
        self$lnorm1 <- nn_layer_norm(embedding_dim + 1)
        self$lnorm2 <- nn_layer_norm(embedding_dim + 1)
        self$linear1 <- nn_linear(embedding_dim + 1, 4 * (embedding_dim + 1))
        self$linear2 <- nn_linear(4 * (embedding_dim + 1), (embedding_dim + 1))
        self$mlp1 <- mlp(length(cardinalities) * (embedding_dim + 1) + num_numerical, fc_units)
        # device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"
        # self$to(device = device)
        self
    },
    forward = function(xcat, xnum, xcoverage) {
        xcat_out <- model$col_embedder(xcat)
        shapes <- xcat_out$shape
        embedded_reshape <- xcat_out$view(list(shapes[2], shapes[1], shapes[3]))

        attn = model$attn(embedded_reshape, embedded_reshape, embedded_reshape)[[2]]
        embedded_reshape <- model$attn(embedded_reshape, embedded_reshape, embedded_reshape)[[1]] + embedded_reshape
        embedded_reshape <- model$lnorm1(embedded_reshape)
        xcat_out_a <- embedded_reshape %>%
            model$linear1() %>%
            nnf_relu() %>%
            model$linear2()
        embedded_reshape <- model$lnorm2(embedded_reshape + xcat_out_a)
        embedded_reshape <- embedded_reshape$view(list(shapes[1], shapes[2]*shapes[3]))

        concat <- torch_cat(list(embedded_reshape, xnum), dim = 2)
        ratio <- self$mlp1(concat) %>%
            nnf_sigmoid()
        ratio * xcoverage
    }
)

rmse <- function(actuals, preds) {
    sqrt(sum((actuals - preds)^2) / length(actuals))
}

mae <- function(actuals, preds) {
    mean(abs(actuals - preds))
}

mean_gamma_deviance <- function(actuals, preds) {
    epsilon <- .Machine$double.eps
    actuals <- actuals + epsilon
    preds <- preds + epsilon
    2 * mean((actuals - preds) / preds - log(actuals / preds))
}
