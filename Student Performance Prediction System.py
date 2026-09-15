# ============================================================
# STUDENT PERFORMANCE PREDICTION SYSTEM
# Machine Learning Project using Python
# ============================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

# Sample dataset creation
data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 1, 2, 9],
    'Attendance': [60, 65, 70, 75, 80, 85, 90, 55, 58, 95],
    'Assignments_Completed': [3, 4, 5, 6, 7, 8, 9, 2, 3, 10],
    'Previous_Score': [45, 50, 55, 60, 65, 70, 75, 40, 42, 90],
    'Final_Score': [48, 52, 58, 65, 70, 78, 85, 40, 45, 95]
}

df = pd.DataFrame(data)

# Display Dataset
print("\n========== DATASET ==========\n")
print(df)

# ============================================================
# STEP 2: DATA VISUALIZATION
# ============================================================

plt.figure(figsize=(8,5))
sns.scatterplot(x='Hours_Studied', y='Final_Score', data=df)
plt.title("Hours Studied vs Final Score")
plt.show()

# ============================================================
# STEP 3: FEATURE SELECTION
# ============================================================

X = df[['Hours_Studied', 'Attendance',
        'Assignments_Completed', 'Previous_Score']]

y = df['Final_Score']

# ============================================================
# STEP 4: SPLIT DATASET
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# STEP 5: TRAIN MODEL
# ============================================================

model = LinearRegression()

model.fit(X_train, y_train)

# ============================================================
# STEP 6: PREDICTION
# ============================================================

y_pred = model.predict(X_test)

print("\n========== PREDICTED SCORES ==========\n")
print(y_pred)

# ============================================================
# STEP 7: MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========\n")

print("Mean Absolute Error :", mae)
print("Mean Squared Error  :", mse)
print("Root Mean Squared Error :", rmse)
print("R2 Score :", r2)

# ============================================================
# STEP 8: USER INPUT PREDICTION
# ============================================================

print("\n========== STUDENT PERFORMANCE PREDICTION ==========\n")

hours = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance Percentage: "))
assignments = float(input("Enter Assignments Completed: "))
previous = float(input("Enter Previous Score: "))

student_data = [[hours, attendance, assignments, previous]]

predicted_score = model.predict(student_data)

print("\nPredicted Final Score =", round(predicted_score[0], 2))

# ============================================================
# STEP 9: RESULT ANALYSIS
# ============================================================

if predicted_score >= 80:
    print("Performance: Excellent")
elif predicted_score >= 60:
    print("Performance: Good")
elif predicted_score >= 40:
    print("Performance: Average")
else:
    print("Performance: Poor")

# ============================================================
# END OF PROJECT
# ============================================================