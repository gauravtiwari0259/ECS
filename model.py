import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('microgrid_training_data_obscured.csv')


if 'target' in df.columns:
    target_col = 'target'
elif 'label' in df.columns:
    target_col = 'label'
else:

    target_col = df.columns[-1]

df = df[df[target_col] != 2]

X = df[['hour_sin', 'hour_cos', 'solar_v', 'batt_v', 'load_i']]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective='binary:logistic'
)

print("Training 2-state microgrid XGBoost model...")
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
print(classification_report(y_test, preds))

model.save_model('microgrid_xgboost_model.json')
print("New 2-class model saved successfully as 'microgrid_xgboost_model.json'")