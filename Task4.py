import numpy as np
import matplotlib.pyplot as plt

# set paramett
gamma = 2.0
beta = 0.985**20
R = 1.025**20
productivity_levels = np.array([0.8027, 1.0, 1.2457])
P = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1360],
    [0.0021, 0.2528, 0.7451]
])
asset_grid = np.linspace(0.01, 10.0, 100) 
n_prod = len(productivity_levels)
n_grid = len(asset_grid)
tax_rate = 0.30
population_ratio_young_age = np.array([1/3, 1/3, 1/3])

# define utility function
def util(c, gamma):
    return (c**(1 - gamma)) / (1 - gamma)

# solve function
def solve_model_and_get_value_functions(pension_amount, tax_rate_middle_age):
    value_t3 = np.array([util(a + pension_amount, gamma) for a in asset_grid])

    value_t2 = np.zeros((n_prod, n_grid))
    policy_t2 = np.zeros((n_prod, n_grid))
    for prod_idx in range(n_prod):
        labor_income_after_tax = productivity_levels[prod_idx] * (1 - tax_rate_middle_age)
        for i, a in enumerate(asset_grid):
            savings_options = asset_grid.copy()
            consumption_options = a + labor_income_after_tax - savings_options
            feasible_indices = np.where(consumption_options >= 0)
            consumption_feasible = consumption_options[feasible_indices]
            savings_feasible = savings_options[feasible_indices]
            current_utility = util(consumption_feasible, gamma)

            next_asset_grid = R * savings_feasible
            next_value_interp = np.interp(next_asset_grid, asset_grid, value_t3)
            expected_next_value = np.zeros_like(next_value_interp)
            for next_prod_idx in range(n_prod):
                expected_next_value += P[prod_idx, next_prod_idx] * next_value_interp
            total_utility = current_utility + beta * expected_next_value
            
            max_idx = np.argmax(total_utility)
            value_t2[prod_idx, i] = total_utility[max_idx]
            policy_t2[prod_idx, i] = savings_feasible[max_idx]

    value_t1 = np.zeros((n_prod, n_grid))
    policy_t1 = np.zeros((n_prod, n_grid))
    for prod_idx in range(n_prod):
        labor_income = productivity_levels[prod_idx]
        for i, a in enumerate(asset_grid):
            savings_options = asset_grid.copy()
            consumption_options = a + labor_income - savings_options
            feasible_indices = np.where(consumption_options >= 0)
            consumption_feasible = consumption_options[feasible_indices]
            savings_feasible = savings_options[feasible_indices]
            current_utility = util(consumption_feasible, gamma)
            
            next_asset_grid = R * savings_feasible
            expected_next_value = np.zeros_like(next_asset_grid)
            for next_prod_idx in range(n_prod):
                next_value_interp = np.interp(next_asset_grid, asset_grid, value_t2[next_prod_idx, :])
                expected_next_value += P[prod_idx, next_prod_idx] * next_value_interp
            total_utility = current_utility + beta * expected_next_value
            
            max_idx = np.argmax(total_utility)
            value_t1[prod_idx, i] = total_utility[max_idx]
            policy_t1[prod_idx, i] = savings_feasible[max_idx]
            
    return value_t1, policy_t1

# pension per person
prob_prod_middle_age = population_ratio_young_age @ P
total_government_tax_revenue = np.sum(productivity_levels * tax_rate * prob_prod_middle_age)
goverment_managed_asset = total_government_tax_revenue * R
pension_per_person = goverment_managed_asset / np.sum(population_ratio_young_age)

print(f"pension per person: {pension_per_person:.4f}\n")

# comparison
value_t1_no_pension, _ = solve_model_and_get_value_functions(pension_amount=0.0, tax_rate_middle_age=0.0)
expected_lifetime_utility_no_pension_by_prod = value_t1_no_pension[:, 0]
overall_expected_lifetime_utility_no_pension = np.sum(expected_lifetime_utility_no_pension_by_prod * population_ratio_young_age)
print(f"Expected Youth Utility without Pension: {overall_expected_lifetime_utility_no_pension:.4f}")

value_t1_with_pension, _ = solve_model_and_get_value_functions(pension_amount=pension_per_person, tax_rate_middle_age=tax_rate)

expected_lifetime_utility_with_pension_by_prod = value_t1_with_pension[:, 0]
overall_expected_lifetime_utility_with_pension = np.sum(expected_lifetime_utility_with_pension_by_prod * population_ratio_young_age)
print(f"Expected Youth Utility with Pension: {overall_expected_lifetime_utility_with_pension:.4f}")


if overall_expected_lifetime_utility_with_pension > overall_expected_lifetime_utility_no_pension:
    print("\nOverall expected lifetime utility will increase with pension.")
elif overall_expected_lifetime_utility_with_pension < overall_expected_lifetime_utility_no_pension:
    print("\nOverall expected lifetime utility will decrease with pension.")
else:
    print("\nOverall expected lifetime utility will stay the same with pension.")
