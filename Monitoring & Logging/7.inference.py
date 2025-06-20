import joblib
import pandas as pd

def load_model(model_path):
    return joblib.load(model_path)

def predict(model, input_data):
    return model.predict(input_data)

if __name__ == "__main__":
    # Raw input (before encoding)
    raw_input = pd.DataFrame({
        "study_hours_per_day": [3],
        "social_media_hours": [2],
        "netflix_hours": [1],
        "attendance_percentage": [95],
        "sleep_hours": [7],
        "exercise_frequency": [3],
        "mental_health_rating": [4],
        "gender": ["Male"],
        "part_time_job": ["No"],
        "diet_quality": ["Good"],
        "parental_education_level": ["Master"],
        "internet_quality": ["Good"],
        "extracurricular_participation": ["Yes"]
    })

    # Apply the same encoding as in training
    input_data = pd.get_dummies(raw_input)

    # Load model
    model = load_model("model.joblib")

    # Ensure columns match training columns
    if hasattr(model, "feature_names_in_"):
        input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
    else:
        # If model.feature_names_in_ is not available, manually specify columns as used in training
        pass  # Add your column list here if needed

    prediction = predict(model, input_data)
    print(f"Prediction: {prediction}")