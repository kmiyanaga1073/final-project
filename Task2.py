import numpy as np

# set parameters
R = 1.025**20
productivity_levels = np.array([0.8027, 1.0, 1.2457])
n_prod = len(productivity_levels)
population_ratio_young_age = np.array([1/3, 1/3, 1/3])
P = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1360],
    [0.0021, 0.2528, 0.7451]
])
tax_rate_middle_age = 0.30

# calculate distribution
prob_prod_t1 = population_ratio_young_age
prob_prod_middle_age = prob_prod_t1 @ P
print(f"Youth Productivity Distribution: {prob_prod_t1}")
print(f"Mid-Age Productivity Distribution: {prob_prod_middle_age}")

# calculate tax revenue
taxable_income_by_prod_type = productivity_levels * tax_rate_middle_age
total_government_tax_revenue_middle_age = np.sum(taxable_income_by_prod_type * prob_prod_middle_age)
print(f"\nMid-Age Government Tax Revenue: {total_government_tax_revenue_middle_age:.4f}")

# asset management
government_managed_asset_at_old_age = total_government_tax_revenue_middle_age * R
print(f"Old-Age Government Asset: {government_managed_asset_at_old_age:.4f}")

# pension per person
pension_per_person = government_managed_asset_at_old_age / np.sum(population_ratio_young_age)
print(f"Pension per Person: {pension_per_person:.4f}")
