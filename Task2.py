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

# -------------------------------------------------------------
# 6. 経済学的直感の記述（コメントとして）
# -------------------------------------------------------------
"""
経済学的直感:

1.  **税収の源泉**: 政府の税収は、中年期の労働者の所得から徴収されます。
    そのため、中年期にどの生産性レベルの労働者がどれだけの割合で存在するか（中年期の生産性分布）が税収額を決定する重要な要素となります。
    若年期の初期分布と遷移行列Pによって、この中年期の分布が定まります。

2.  **年金額の決定**:
    * **税収**: 当然ながら、より多くの税収が徴収されれば、より多くの年金が給付されます。
    * **利子率**: 徴収された税収は政府によって運用されるため、利子率が高ければ高いほど、年金として給付できる総額が増加し、結果として一人当たりの年金額も増加します。
    * **人口**: 年金は全人口に均等に給付されるため、給付対象となる人口が少なければ一人当たりの額は増加し、多ければ減少します（このモデルでは人口は標準化されているため、その影響は直接見えませんが、実世界では重要です）。

このシミュレーション結果は、年金制度の持続可能性を考える上で、税収を支える経済活動（労働所得）と、その税収をいかに効率的に運用するかが鍵であることを示唆します。
"""