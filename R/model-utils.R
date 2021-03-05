flood_dataset <- dataset(
    "flood",
    initialize = function(df, categorical_cols, numeric_cols, response_col = NULL, env = NULL) {
        self$is_train <- if (!is.null(response_col)) TRUE else FALSE
        self$xcat <- df[categorical_cols] %>%
            as.matrix() %>%
            `+`(1L) %>%
            torch_tensor()
        if (!is.null(env)) {
            assign("cardinalities", sapply(df[categorical_cols], function(x) max(x) + 2), envir = env)
        }
        self$xnum <- df[numeric_cols] %>%
            as.matrix() %>%
            torch_tensor()

        if (self$is_train) {
            self$y <- df[[response_col]] %>%
                as.matrix() %>%
                torch_tensor()
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
    initialize = function(cardinalities, fn_embedding_dim) {
        self$embeddings <- nn_module_list(lapply(
            cardinalities,
            function(x) {
                nn_embedding(
                    num_embeddings = x, embedding_dim = fn_embedding_dim(x),
                    max_norm = 10, norm_type = 2
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

net <- nn_module(
    "baseline_net",
    initialize = function(cardinalities,
                          num_numerical,
                          fn_embedding_dim = function(x) ceiling(x / 2)) {
        self$embedder <- embedding_module(cardinalities, fn_embedding_dim)
        sum_embedding_dim <- sapply(cardinalities, fn_embedding_dim) %>%
            sum()
        self$output <- nn_linear(sum_embedding_dim + num_numerical, 1)
    },
    forward = function(xcat, xnum) {
        embedded <- self$embedder(xcat)
        all <- torch_cat(list(embedded, xnum$to(dtype = torch_float())), dim = 2)
        all %>%
            self$output() %>%
            nnf_softplus()
    }
)

train_loop <- function(model, train_dl, valid_dl, epochs, optimizer) {
    for (epoch in seq_len(epochs)) {
        model$train()
        train_losses <- c()

        for (b in enumerate(train_dl)) {
            optimizer$zero_grad()
            output <- model(b$x[[1]], b$x[[2]])
            loss <- nnf_mse_loss(output, b$y)
            loss$backward()
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

get_preds <- function(model, dl) {
    preds <- numeric(0)
    for (b in enumerate(dl)) {
        preds <- c(preds, model(b$x[[1]], b$x[[2]]) %>% as.array())
    }
    preds
}