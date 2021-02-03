library(tabnet)
library(tidyverse)
library(recipes)
library(rsample)

# Download data
claims <- cellar::cellar_pull("nfip_claims") %>%
  filter(year_of_loss > 2000)

# Basic feature engineering
small_data <- claims %>%
  filter(amount_paid_on_building_claim > 0,
         total_building_insurance_coverage > 0) %>%
  sample_n(10000) %>%
  select(amount_paid_on_building_claim, total_building_insurance_coverage,
         reported_zip_code, state, primary_residence, basement_enclosure_crawlspace_type,
         condominium_indicator, number_of_floors_in_the_insured_building, occupancy_type,
         community_rating_system_discount, flood_zone)


rec <- recipe(
    amount_paid_on_building_claim ~ total_building_insurance_coverage + community_rating_system_discount +
        basement_enclosure_crawlspace_type + number_of_floors_in_the_insured_building + flood_zone + occupancy_type,
    data = small_data
) %>%
    step_normalize(all_numeric(), -all_outcomes()) %>%
    step_unknown(all_nominal(), -all_outcomes())

fit <- tabnet_fit(rec, small_data, epochs = 1)

# example: extract embedding weights
fit$fit$network$parameters$embedder.embeddings.2.weight

# example: prediction
predict(fit, small_data %>% sample_n(100))

ex_fit <- tabnet_explain(fit, small_data)

# example: attention masks visualization
ex_fit$masks %>% 
  imap_dfr(~mutate(
    .x, 
    step = sprintf("Step %d", .y),
    rowname = row_number()
  )) %>% 
  pivot_longer(-c(rowname, step), names_to = "variable", values_to = "m_agg") %>% 
  ggplot(aes(x = rowname, y = variable, fill = m_agg)) +
  geom_tile() +
  scale_fill_viridis_c() +
  facet_wrap(~step)
