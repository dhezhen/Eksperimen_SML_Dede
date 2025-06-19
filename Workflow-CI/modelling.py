import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow

# Tracking otomatis
mlflow.sklearn.autolog()

# Muat data (ganti sesuai hasil preprocessing)
df = pd.read_csv("../preprocessing/student_habit_performance_preprocessing.csv")
# Misal: kolom target 'Status'
X = df.drop(columns=['exam_score'])
y = df['exam_score']  # Ganti dengan nama kolom target yang sesuai


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mulai eksperimen
with mlflow.start_run():
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
