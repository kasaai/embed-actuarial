claims <- cellar::cellar_pull("nfip_claims") %>%
    filter(year_of_loss > 2000)

small_data <- claims %>%
    filter(
        amount_paid_on_building_claim > 0,
        total_building_insurance_coverage > 0
    ) %>%
    sample_n(10000) %>%
    select(
        amount_paid_on_building_claim, total_building_insurance_coverage,
        reported_zip_code, state, primary_residence, basement_enclosure_crawlspace_type,
        condominium_indicator, number_of_floors_in_the_insured_building, occupancy_type,
        community_rating_system_discount, flood_zone
    ) %>%
    filter(across(everything(), ~!is.na(.x))) %>% 
    mutate(loss_proportion = pmin(amount_paid_on_building_claim / total_building_insurance_coverage, 1))
