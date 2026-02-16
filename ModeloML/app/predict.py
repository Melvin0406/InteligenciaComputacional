import joblib
import pandas as pd
from pathlib import Path

# Cargar el modelo exportado (pipeline completo: preprocesamiento + modelo)
script_dir = Path(__file__).resolve().parent
model = joblib.load(script_dir / ".." / "model.joblib")

# Datos de entrada hardcodeados (ejemplo de un estudiante)
student_data = pd.DataFrame([{
    "age": 21,
    "gender": "Male",
    "study_hours_per_day": 5.0,
    "sleep_hours": 7.0,
    "phone_usage_hours": 3.0,
    "social_media_hours": 1.5,
    "youtube_hours": 1.0,
    "gaming_hours": 2.0,
    "breaks_per_day": 4,
    "coffee_intake_mg": 200,
    "exercise_minutes": 30,
    "assignments_completed": 8,
    "attendance_percentage": 85.0,
    "stress_level": 5,
    "focus_score": 65,
    "final_grade": 78.0
}])

# Realizar predicción
prediction = model.predict(student_data)

print("=" * 50)
print("  Predicción de Productividad Estudiantil")
print("=" * 50)
print()
print("Datos del estudiante:")
for col in student_data.columns:
    print(f"  {col}: {student_data[col].values[0]}")
print()
print(f"Productivity Score predicho: {prediction[0]:.2f}")
print("=" * 50)
