import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv('microgrid_training_data_obscured.csv')

X = df.drop('target_action', axis=1)
y = df['target_action']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Solar (0)', 'Battery (1)', 'Shed Load (2)']))

model.save_model('microgrid_xgboost_model.json')
print("Model saved to 'microgrid_xgboost_model.json'. Ready for FastAPI integration!")