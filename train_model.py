import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Generate Synthetic Dataset (Replace with real CSV like UCI Student Performance)
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'attendance_rate': np.random.uniform(50, 100, n_samples),
    'study_hours_per_week': np.random.uniform(2, 30, n_samples),
    'previous_gpa': np.random.uniform(1.5, 4.0, n_samples),
    'assignments_completed': np.random.uniform(40, 100, n_samples),
    'lms_logins_per_month': np.random.randint(5, 100, n_samples),
    'extracurricular': np.random.choice(['Yes', 'No'], n_samples),
    'internet_access': np.random.choice(['Yes', 'No'], n_samples, p=[0.85, 0.15])
})

# Define target logic (1 = Success/Pass, 0 = At-Risk/Fail)
score = (
    0.3 * data['attendance_rate'] +
    0.8 * data['study_hours_per_week'] +
    10.0 * data['previous_gpa'] +
    0.2 * data['assignments_completed'] +
    np.random.normal(0, 5, n_samples)
)
data['success'] = (score > np.percentile(score, 30)).astype(int)

X = data.drop('success', axis=1)
y = data['success']

# 2. Define Preprocessing
numeric_features = ['attendance_rate', 'study_hours_per_week', 'previous_gpa', 'assignments_completed', 'lms_logins_per_month']
categorical_features = ['extracurricular', 'internet_access']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# 3. Build Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# 4. Train and Evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test)[:, 1]

print("--- Classification Report ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# 5. Save Model
joblib.dump(model_pipeline, 'student_success_model.pkl')
print("Model saved to student_success_model.pkl")
