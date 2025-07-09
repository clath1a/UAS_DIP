# app.py (dengan tambahan kode debug)
import os
import json
import cv2
import numpy as np
import joblib
from flask import Flask, request, render_template, url_for, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from skimage.feature import local_binary_pattern

# Inisialisasi dan Konfigurasi
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_ROOT, 'models')
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'uploads')
DATA_FOLDER = os.path.join(APP_ROOT, 'data')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# Fungsi Bantuan
def load_json_data(filename):
    try:
        with open(os.path.join(DATA_FOLDER, filename), 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        # --- DEBUG 1: Menangkap error jika file JSON gagal dimuat ---
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR saat memuat {filename}: {e}")
        print(f"Pastikan file ada di folder 'data' dan isinya valid.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return {}

def extract_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_sat = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    cv2.normalize(hist_hue, hist_hue); cv2.normalize(hist_sat, hist_sat)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 3; n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist_lbp = hist_lbp.astype("float"); hist_lbp /= (hist_lbp.sum() + 1e-6)
    return np.hstack([hist_hue.flatten(), hist_sat.flatten(), hist_lbp])

def classify_rice_by_rules(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv_image)
    avg_hue, avg_saturation, avg_value = mean_hsv[0], mean_hsv[1], mean_hsv[2]
    if avg_value < 75: return "Ketan Hitam"
    if avg_saturation < 45 and avg_value > 150: return "Putih"
    if (avg_hue <= 15 or avg_hue >= 165) and avg_saturation > 45: return "Merah"
    if avg_saturation > 45 and avg_value >= 75: return "Merah"
    return "Tidak Diketahui"

# Muat Data
fun_facts = load_json_data('fun_facts.json')
model_performance = load_json_data('model_performance.json')
# --- DEBUG 2: Memeriksa data performa setelah dimuat ---
print("--- [DEBUG] Data Kinerja Model yang berhasil dimuat: ---")
print(model_performance)
print("-------------------------------------------------------")

MODEL_MAPPING = {'svm': 'model_svm_beras.pkl','rule_based': 'model_rule_based.pkl','ann': 'model_ann_beras_regularized.h5'}

# Routing
@app.route('/')
def index():
    return render_template('index.html', error=None, models=MODEL_MAPPING.keys())

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('index.html', error='Tidak ada file yang dipilih.', models=MODEL_MAPPING.keys())
    file = request.files['file']
    model_choice = request.form.get('model_selection')

    # --- DEBUG 3: Memeriksa model yang dipilih pengguna ---
    print(f"\n--- [DEBUG] Pengguna memilih model: '{model_choice}' ---")

    if not model_choice or model_choice not in MODEL_MAPPING:
        return render_template('index.html', error='Pilihan model tidak valid.', models=MODEL_MAPPING.keys())
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        img = cv2.imread(filepath)
        if img is None: return render_template('index.html', error='Format gambar tidak didukung.', models=MODEL_MAPPING.keys())
        img_resized = cv2.resize(img, (100, 100))
        prediction_label = ""
        
        if model_choice == 'rule_based': prediction_label = classify_rice_by_rules(img_resized)
        elif model_choice == 'ann':
            model = tf.keras.models.load_model(os.path.join(MODEL_DIR, MODEL_MAPPING['ann']))
            scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_ann.pkl'))
            le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder_ann.pkl'))
            features = extract_features(img_resized).reshape(1, -1)
            features_scaled = scaler.transform(features)
            prediction_idx = np.argmax(model.predict(features_scaled), axis=1)
            prediction_label = le.inverse_transform(prediction_idx)[0]
        elif model_choice == 'svm':
            model = joblib.load(os.path.join(MODEL_DIR, 'model_svm_beras.pkl'))
            scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
            le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
            features = extract_features(img_resized).reshape(1, -1)
            features_scaled = scaler.transform(features)
            prediction_idx = model.predict(features_scaled)
            prediction_label = le.inverse_transform(prediction_idx)[0]

        facts = fun_facts.get(prediction_label, ["Fakta menarik tidak ditemukan."])
        performance_data = model_performance.get(model_choice, {})
        accuracy = performance_data.get('accuracy', 'N/A')
        report_dict = performance_data.get('report', {})
        report_list = [{'class': k, 'precision': v['precision'], 'recall': v['recall'], 'f1_score': v['f1-score'], 'support': v['support']} for k, v in report_dict.items()]
        confusion_matrix_url = url_for('static', filename=f'images/confusion_matrix_{model_choice}.png')
        
        # --- DEBUG 4: Memeriksa data final sebelum dikirim ke template ---
        print("--- [DEBUG] Data final yang akan dikirim ke template: ---")
        print(f"  > Accuracy: {accuracy}")
        print(f"  > Jumlah baris di report_list: {len(report_list)}")
        print(f"  > URL Confusion Matrix: {confusion_matrix_url}")
        print("------------------------------------------------------\n")

        return render_template('result.html', prediction=prediction_label, facts=facts, image_url=url_for('uploaded_file', filename=filename), model_name=model_choice, accuracy=accuracy, report=report_list, confusion_matrix=confusion_matrix_url)
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR saat prediksi: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return render_template('index.html', error=f'Terjadi kesalahan: {e}', models=MODEL_MAPPING.keys())

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)