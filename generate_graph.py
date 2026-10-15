import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

model = xgb.XGBClassifier()
model.load_model('microgrid_xgboost_model.json')

df = pd.read_csv('microgrid_training_data_obscured.csv')
X = df[['hour_sin', 'hour_cos', 'solar_v', 'batt_v', 'load_i']].head(200).copy()

np.random.seed(42)
X['batt_v'] += np.random.normal(0, 0.08, size=len(X))

def naive_rule_controller(batt_v, solar_v):
    if solar_v < 2.0 or batt_v < 12.15:
        return 1
    return 0

rule_preds = [naive_rule_controller(row['batt_v'], row['solar_v']) for _, row in X.iterrows()]

X_model = X[['hour_sin', 'hour_cos', 'solar_v', 'batt_v', 'load_i']]
xgb_preds = model.predict(X_model)

plt.figure(figsize=(12, 5))
plt.plot(rule_preds, label='Traditional Rule-Based (High Chattering)', color='#EF5350', alpha=0.7, linestyle='--', linewidth=2)
plt.plot(xgb_preds, label='XGBoost AI Controller (Stable State Estimation)', color='#29B6F6', linewidth=2.5)

plt.title('Relay Actuation Stability: Rule-Based vs. XGBoost AI Controller', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Operational Time Steps (Stress-Tested with Noise)', fontsize=12)
plt.ylabel('Relay State (0: Solar | 1: Battery)', fontsize=12)
plt.yticks([0, 1], ['Solar Mode', 'Battery Mode'])
plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.savefig('switching_comparison.png', dpi=300)
print("Graph generated and saved successfully as 'switching_comparison.png'!")
plt.show()