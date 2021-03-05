library(tidyverse)
library(recipes)
library(rsample)
library(torch)

source("R/model-utils.R")
source("R/data-loading.R")

categorical_cols <- c(
    "state", "primary_residence", "basement_enclosure_crawlspace_type",
    "condominium_indicator", "number_of_floors_in_the_insured_building",
    "occupancy_type", "flood_zone"
)

numeric_cols <- c("total_building_insurance_coverage", "community_rating_system_discount")
response_col <- "amount_paid_on_building_claim"

env <- new.env()

rec <- recipe(amount_paid_on_building_claim ~ .,
   data = small_data %>%
      select(-loss_proportion, -reported_zip_code)
) %>%
   step_normalize(all_numeric(), -all_outcomes()) %>%
   step_unknown(all_nominal()) %>%
   step_integer(all_nominal(), strict = TRUE, zero_based = TRUE) %>%
   prep()

glimpse(small_data)

training_data <- juice(rec)

ds <- flood_dataset(training_data, categorical_cols, numeric_cols, response_col, env)

model <- net(env$cardinalities, length(numeric_cols), function(x) 2)

train_dl <- ds %>% dataloader(batch_size = 1024, shuffle = TRUE)

o1 <- model(ds[1:10]$x[[1]], ds[1:10]$x[[2]])
o1
y1 <- ds[1:10]$y
y1
optimizer <- optim_adam(model$parameters, lr = 0.1)

train_loop(model, train_dl, train_dl, epochs = 2, optimizer = optimizer)

ds
test_ds <- flood_dataset(training_data, categorical_cols, numeric_cols) 
test_dl <- test_ds %>% dataloader(batch_size = 1024, shuffle = FALSE)
foo <- enumerate(test_dl)
preds <- numeric(0)
for (b in enumerate(test_dl))  {
    preds <- c(preds, model(b$x[[1]], b$x[[2]]) %>% as.array())
}
preds
length(preds)
