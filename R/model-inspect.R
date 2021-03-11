

a <- broom::tidy(cv_results[[1]]$model_glm, exponentiate = TRUE)
View(a)
  # perform initial tidying of model
library(broom.helpers)
a <- cv_results[[1]]$model_glm %>% 
  tidy_and_attach(exponentiate = TRUE) %>%
  # add reference row
  tidy_add_reference_rows() %>%
  # add term labels
  tidy_add_term_labels() %>%
  # remove intercept
  tidy_remove_intercept
View(a)
b <- a %>%
    arrange(var_type) %>%
    filter(variable %in% c("occupancy_type", "flood_zone")) %>%
    select(variable, label, estimate) %>%
    mutate(estimate = ifelse(is.na(estimate), 1, estimate)) %>%
    rename(relativity = estimate)
View(b)
b %>% knitr::kable(digits = 2)

model <- cv_results[[1]]$model_nn
rec <- cv_results[[1]]$rec_nn
