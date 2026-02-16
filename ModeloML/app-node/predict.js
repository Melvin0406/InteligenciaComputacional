const ort = require("onnxruntime-node");
const path = require("path");

// Columnas numéricas (mismo orden que en el entrenamiento)
const numCols = [
  "age", "study_hours_per_day", "sleep_hours", "phone_usage_hours",
  "social_media_hours", "youtube_hours", "gaming_hours", "breaks_per_day",
  "coffee_intake_mg", "exercise_minutes", "assignments_completed",
  "attendance_percentage", "stress_level", "focus_score", "final_grade"
];

async function predict(input) {
  console.log("Cargando modelo ONNX...");
  const modelPath = path.join(__dirname, "..", "model.onnx");
  const session = await ort.InferenceSession.create(modelPath);

  // Crear tensores de entrada: cada columna es un tensor individual
  const feeds = {};
  for (const col of numCols) {
    feeds[col] = new ort.Tensor("float32", [input[col]], [1, 1]);
  }
  feeds["gender"] = new ort.Tensor("string", [input.gender], [1, 1]);

  // Ejecutar inferencia
  const results = await session.run(feeds);
  const outputName = session.outputNames[0];
  const prediction = results[outputName].data[0];

  return prediction;
}

// Datos de entrada hardcodeados (ejemplo de un estudiante)
const studentData = {
  age: 21,
  gender: "Male",
  study_hours_per_day: 5.0,
  sleep_hours: 7.0,
  phone_usage_hours: 3.0,
  social_media_hours: 1.5,
  youtube_hours: 1.0,
  gaming_hours: 2.0,
  breaks_per_day: 4,
  coffee_intake_mg: 200,
  exercise_minutes: 30,
  assignments_completed: 8,
  attendance_percentage: 85.0,
  stress_level: 5,
  focus_score: 65,
  final_grade: 78.0,
};

predict(studentData).then((score) => {
  console.log("==================================================");
  console.log("  Predicción de Productividad Estudiantil (ONNX)");
  console.log("==================================================");
  console.log();
  console.log("Datos del estudiante:");
  for (const [key, value] of Object.entries(studentData)) {
    console.log(`  ${key}: ${value}`);
  }
  console.log();
  console.log(`Productivity Score predicho: ${score.toFixed(2)}`);
  console.log("==================================================");
}).catch((err) => {
  console.error("Error:", err.message);
});
