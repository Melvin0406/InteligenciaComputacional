import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
from pathlib import Path

# Cargar modelo
script_dir = Path(__file__).resolve().parent
model = joblib.load(script_dir / ".." / "model.joblib")

# Definición de campos: (nombre_columna, label, tipo, min, max, default)
FIELDS = [
    ("age",                    "Edad",                      "int",   17,   29,   21),
    ("gender",                 "Género",                    "choice", None, None, "Male"),
    ("study_hours_per_day",    "Horas de estudio/día",      "float", 0.5,  10,   5.0),
    ("sleep_hours",            "Horas de sueño",            "float", 3,    10,   7.0),
    ("phone_usage_hours",      "Horas uso de teléfono",     "float", 0.5,  12,   3.0),
    ("social_media_hours",     "Horas redes sociales",      "float", 0,    8,    1.5),
    ("youtube_hours",          "Horas YouTube",             "float", 0,    6,    1.0),
    ("gaming_hours",           "Horas gaming",              "float", 0,    6,    2.0),
    ("breaks_per_day",         "Descansos por día",         "int",   1,    14,   4),
    ("coffee_intake_mg",       "Cafeína (mg)",              "int",   0,    500,  200),
    ("exercise_minutes",       "Minutos de ejercicio",      "int",   0,    120,  30),
    ("assignments_completed",  "Tareas completadas",        "int",   0,    19,   8),
    ("attendance_percentage",  "Asistencia (%)",            "float", 40,   100,  85.0),
    ("stress_level",           "Nivel de estrés (1-10)",    "int",   1,    10,   5),
    ("focus_score",            "Focus score (30-99)",       "int",   30,   99,   65),
    ("final_grade",            "Calificación final",        "float", 40,   100,  78.0),
]

GENDER_OPTIONS = ["Male", "Female", "Other"]


class PredictApp:
    def __init__(self, root):
        root.title("Predicción de Productividad Estudiantil")
        root.resizable(False, False)

        main_frame = ttk.Frame(root, padding=15)
        main_frame.pack()

        ttk.Label(main_frame, text="Predicción de Productividad Estudiantil",
                  font=("Segoe UI", 14, "bold")).grid(row=0, column=0, columnspan=3, pady=(0, 10))

        self.entries = {}

        for i, (col, label, ftype, fmin, fmax, default) in enumerate(FIELDS, start=1):
            ttk.Label(main_frame, text=label).grid(row=i, column=0, sticky="w", padx=(0, 10), pady=2)

            if ftype == "choice":
                var = tk.StringVar(value=default)
                widget = ttk.Combobox(main_frame, textvariable=var, values=GENDER_OPTIONS,
                                      state="readonly", width=18)
                widget.grid(row=i, column=1, sticky="w", pady=2)
                self.entries[col] = var
            else:
                var = tk.StringVar(value=str(default))
                entry = ttk.Entry(main_frame, textvariable=var, width=20)
                entry.grid(row=i, column=1, sticky="w", pady=2)
                range_text = f"({fmin} - {fmax})"
                ttk.Label(main_frame, text=range_text, foreground="gray").grid(row=i, column=2, sticky="w", padx=(5, 0))
                self.entries[col] = var

        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=len(FIELDS) + 1, column=0, columnspan=3, pady=(15, 5))

        ttk.Button(btn_frame, text="Predecir", command=self.predict).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Limpiar", command=self.reset).pack(side="left", padx=5)

        self.result_var = tk.StringVar(value="")
        self.result_label = ttk.Label(main_frame, textvariable=self.result_var,
                                      font=("Segoe UI", 16, "bold"), foreground="#2e7d32")
        self.result_label.grid(row=len(FIELDS) + 2, column=0, columnspan=3, pady=(5, 0))

    def validate_and_collect(self):
        data = {}
        errors = []

        for col, label, ftype, fmin, fmax, default in FIELDS:
            raw = self.entries[col].get().strip()

            if ftype == "choice":
                if raw not in GENDER_OPTIONS:
                    errors.append(f"{label}: selecciona una opción válida")
                else:
                    data[col] = raw
                continue

            if raw == "":
                errors.append(f"{label}: campo vacío")
                continue

            try:
                value = int(raw) if ftype == "int" else float(raw)
            except ValueError:
                expected = "un número entero" if ftype == "int" else "un número"
                errors.append(f"{label}: ingresa {expected}")
                continue

            if value < fmin or value > fmax:
                errors.append(f"{label}: debe estar entre {fmin} y {fmax}")
                continue

            data[col] = value

        return data, errors

    def predict(self):
        data, errors = self.validate_and_collect()

        if errors:
            messagebox.showerror("Errores de validación", "\n".join(errors))
            return

        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        self.result_var.set(f"Productivity Score: {prediction:.2f}")

    def reset(self):
        for col, label, ftype, fmin, fmax, default in FIELDS:
            self.entries[col].set(str(default))
        self.result_var.set("")


if __name__ == "__main__":
    root = tk.Tk()
    PredictApp(root)
    root.mainloop()
