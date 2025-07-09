# app.py
import os
import json
import cv2
import numpy as np
import joblib
from flask import Flask, request, render_template, url_for, send_from_directory
from werkzeug.utils import secure_filename
from skimage.feature import local_binary_pattern
# Import TensorFlow untuk memuat model ANN (.h5)
import tensorflow as tf

# ==============================================================================
# Inisialisasi dan Konfigurasi Aplikasi Flask
# ==============================================================================
app = Flask(__name__)

# Konfigurasi path
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_ROOT, 'models')
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'uploads')
DATA_FOLDER = os.path.join(APP_ROOT, 'data')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan folder-folder yang dibutuhkan ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# ==============================================================================
# Fungsi Bantuan (Helpers)
# ==============================================================================
def load_json_data(filename):
    """Memuat data dari file JSON."""
    try:
        with open(os.path.join(DATA_FOLDER, filename), 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error memuat {filename}: {e}")
        return {}

def extract_features(image):
    """Mengekstrak fitur warna (histogram HSV) dan tekstur (LBP) dari gambar."""
    # Fungsi ini sama untuk model SVM dan ANN
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_sat = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    cv2.normalize(hist_hue, hist_hue)
    cv2.normalize(hist_sat, hist_sat)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-6)

    return np.hstack([hist_hue.flatten(), hist_sat.flatten(), hist_lbp])

# --- PERBAIKAN: Logika untuk Rule-Based dipindahkan langsung ke sini ---
def classify_rice_by_rules(image):
    """Mengklasifikasikan jenis beras berdasarkan aturan warna di ruang HSV."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv_image)
    avg_hue, avg_saturation, avg_value = mean_hsv[0], mean_hsv[1], mean_hsv[2]

    if avg_value < 75: return "Ketan Hitam"
    if avg_saturation < 45 and avg_value > 150: return "Putih"
    if (avg_hue <= 15 or avg_hue >= 165) and avg_saturation > 45: return "Merah"
    if avg_saturation > 45 and avg_value >= 75: return "Merah"
    return "Tidak Diketahui"

# ==============================================================================
# Muat Data
# ==============================================================================
fun_facts = load_json_data('fun_facts.json')

# Peta untuk memilih model berdasarkan input form
MODEL_MAPPING = {
    'svm': 'model_svm_beras.pkl',
    'rule_based': 'model_rule_based.pkl', # Nama ini tetap ada, tapi akan ditangani khusus
    'ann': 'model_ann_beras_regularized.h5'
}

# ==============================================================================
# Routing Aplikasi Flask
# ==============================================================================
@app.route('/', methods=['GET'])
def index():
    """Menampilkan halaman utama untuk upload."""
    # Mengirim daftar nama model ke template
    return render_template('index.html', error=None, models=MODEL_MAPPING.keys())

@app.route('/predict', methods=['POST'])
def predict():
    """Menangani upload gambar dan melakukan prediksi."""
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('index.html', error='Tidak ada file yang dipilih.', models=MODEL_MAPPING.keys())

    file = request.files['file']
    model_choice = request.form.get('model_selection')

    if not model_choice or model_choice not in MODEL_MAPPING:
        return render_template('index.html', error='Pilihan model tidak valid.', models=MODEL_MAPPING.keys())

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        if img is None:
            return render_template('index.html', error='Format gambar tidak didukung.', models=MODEL_MAPPING.keys())
        
        img_resized = cv2.resize(img, (100, 100))
        
        prediction_label = ""
        
        # --- PERBAIKAN: Logika prediksi dipisah berdasarkan pilihan model ---
        
        if model_choice == 'rule_based':
            # Langsung panggil fungsi, tidak perlu load model
            prediction_label = classify_rice_by_rules(img_resized)
            
        elif model_choice == 'ann':
            # Memuat komponen model ANN
            model = tf.keras.models.load_model(os.path.join(MODEL_DIR, MODEL_MAPPING['ann']))
            scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_ann.pkl'))
            le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder_ann.pkl'))
            
            features = extract_features(img_resized).reshape(1, -1)
            features_scaled = scaler.transform(features)
            
            prediction_proba = model.predict(features_scaled)
            prediction_idx = np.argmax(prediction_proba, axis=1)
            prediction_label = le.inverse_transform(prediction_idx)[0]

        elif model_choice == 'svm':
            # Memuat komponen model SVM
            model = joblib.load(os.path.join(MODEL_DIR, MODEL_MAPPING['svm']))
            scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
            le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))

            features = extract_features(img_resized).reshape(1, -1)
            features_scaled = scaler.transform(features)
            
            prediction_idx = model.predict(features_scaled)
            prediction_label = le.inverse_transform(prediction_idx)[0]

        # Dapatkan fakta menarik
        facts = fun_facts.get(prediction_label, ["Fakta menarik tidak ditemukan."])

        return render_template('result.html', 
                               prediction=prediction_label, 
                               facts=facts,
                               image_url=url_for('uploaded_file', filename=filename))
    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return render_template('index.html', error=f'Terjadi kesalahan: {e}', models=MODEL_MAPPING.keys())

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Menyajikan file yang diunggah dari folder 'uploads'."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ==============================================================================
# Menjalankan Aplikasi
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=True)