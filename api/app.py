from flask import Flask, render_template, request, redirect, flash, jsonify
import os
import pickle
import fitz  # PyMuPDF for PDF text extraction
import numpy as np

app = Flask(__name__, template_folder="../templates", static_folder="../static")

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = "your_secret_key"  # For flash messages

# Define paths for the model, vectorizer, and label encoder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "../model/resume_classifier.pkl")
vectorizer_path = os.path.join(BASE_DIR, "../model/tfidf_vectorizer.pkl")
label_encoder_path = os.path.join(BASE_DIR, "../model/label_encoder.pkl")

# Load trained models
try:
    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
    label_encoder = pickle.load(open(label_encoder_path, "rb"))
    print("✅ Model, vectorizer, and label encoder loaded successfully!")

except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit("🚨 Model files not found! Please check the paths and retrain the model if necessary.")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = " ".join([page.get_text("text") for page in doc])
        return text.strip()
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        return ""

# Function to predict all possible job categories
def predict_all_possible_categories(resume_text, threshold=0.05):
    try:
        vectorized_text = vectorizer.transform([resume_text])
        probabilities = model.predict_proba(vectorized_text)[0]

        # Get categories with probability above threshold
        possible_indices = np.where(probabilities >= threshold)[0]
        possible_categories = label_encoder.inverse_transform(possible_indices)

        # Sort by probability (highest first)
        sorted_categories = [cat for _, cat in sorted(zip(probabilities[possible_indices], possible_categories), reverse=True)]

        return list(sorted_categories)
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return ["Unknown"]

# Route for home page (Upload Page)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part", "error")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No selected file", "error")
            return redirect(request.url)

        if file and file.filename.endswith(".pdf"):
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            extracted_text = extract_text_from_pdf(file_path)
            if not extracted_text:
                flash("Could not extract text from PDF", "error")
                return redirect(request.url)

            job_categories = predict_all_possible_categories(extracted_text, threshold=0.05)

            return render_template("result.html", categories=job_categories, resume_text=extracted_text)

        flash("Invalid file format. Please upload a PDF.", "error")
        return redirect(request.url)

    return render_template("index.html")

# API Route for AJAX Requests
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and file.filename.endswith(".pdf"):
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        extracted_text = extract_text_from_pdf(file_path)
        if not extracted_text:
            return jsonify({"error": "Could not extract text from PDF"}), 400

        job_categories = predict_all_possible_categories(extracted_text, threshold=0.05)

        return jsonify({"categories": job_categories})

    return jsonify({"error": "Invalid file format. Please upload a PDF."}), 400

if __name__ == "__main__":
    app.run(debug=True)
