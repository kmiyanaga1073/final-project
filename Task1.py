import numpy as np
import matplotlib.pyplot as plt

# set parameters
gamma = 2.0
beta = 0.985**20
R = 1.025**(20-1)
productivity = np.array([0.8027, 1.0, 1.2457])
n_prod = len(productivity)
P = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1360],
    [0.0021, 0.2528, 0.7451]
])
n_grid = 100
max_asset = 10.0
asset_grid = np.linspace(0.01, max_asset, n_grid)

# define utility function
def util(c, gamma):
    return (c**(1 - gamma)) / (1 - gamma)

# backward induction
value_t3 = util(asset_grid, gamma)
value_t2 = np.zeros((n_prod, n_grid))
policy_t2 = np.zeros((n_prod, n_grid))

for prod_idx in range(n_prod):
    labor_income = productivity[prod_idx]
    for i, a in enumerate(asset_grid):
        savings_options = asset_grid.copy()
        consumption_options = a + labor_income - savings_options
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
    labor_income = productivity[prod_idx]
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

# graph
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(asset_grid, policy_t1[0, :], label='Low Productivity', color='blue', linestyle='-')
ax.plot(asset_grid, policy_t1[1, :], label='Medium Productivity', color='green', linestyle='--')
ax.plot(asset_grid, policy_t1[2, :], label='High Productivity', color='red', linestyle='-.')
ax.plot(asset_grid, asset_grid, 'k--', label='45-degree line')
ax.set_title('Youth Asset Policy Function (No Pension)', fontsize=16)
ax.set_xlabel('Youth Asset without Interest (a)', fontsize=12)
ax.set_ylabel('Youth Asset (a\')', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0, max_asset)
ax.set_xlim(0, max_asset)
plt.show()