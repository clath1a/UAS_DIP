# 1. Import Library
import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Mount Google Drive
try:
    drive.mount('/content/drive')
    drive_mounted = True
except Exception as e:
    print(f"Google Drive tidak dapat di-mount: {e}. Script akan berjalan di lingkungan lokal.")
    drive_mounted = False

# 3. Definisi Path dan Kelas
# Ganti 'Dataset_Beras' dengan nama folder Anda di Google Drive
if drive_mounted:
    base_path = "/content/drive/MyDrive/Dataset_Beras/"
else:
    base_path = "./Dataset_Beras/"

input_path = os.path.join(base_path, "training")
output_path = os.path.join(base_path, "processed")
os.makedirs(output_path, exist_ok=True)
os.makedirs(os.path.join(output_path, "segmented_images_edge"), exist_ok=True)

# Nama kelas disesuaikan dengan jenis beras.
classes = ["beras_merah", "ketan_putih", "IR64", "ketan_hitam", "pandan_wangi"]

# 4. FUNGSI PREPROCESSING MENGGUNAKAN DETEKSI TEPI (EDGE DETECTION)
def preprocess_with_edge_detection(image_path):
    """
    Fungsi segmentasi menggunakan Canny Edge Detection untuk mengisolasi objek beras.
    Metode ini efektif untuk background terang maupun gelap.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Tidak dapat membaca gambar di {image_path}")
        return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Langkah 1: Konversi ke grayscale dan lakukan blurring untuk mengurangi noise
    # Blurring sangat penting untuk mendapatkan hasil deteksi tepi yang bersih.
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Langkah 2: Lakukan Canny Edge Detection
    # Threshold dihitung secara otomatis berdasarkan median gambar untuk membuatnya adaptif.
    v = np.median(blurred)
    sigma = 0.33
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(blurred, lower_thresh, upper_thresh)

    # Langkah 3: Tutup celah pada garis tepi untuk membuat bentuk yang solid
    # Operasi 'closing' akan menyambungkan garis-garis tepi yang terputus.
    kernel = np.ones((5,5),np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Langkah 4: Temukan kontur dari tepi yang sudah solid dan buat mask
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)
    if contours:
        # Asumsikan butir beras adalah kontur dengan area terbesar
        cnt = max(contours, key=cv2.contourArea)
        # Gambar kontur yang ditemukan ke mask dengan warna putih (filled)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # Terapkan mask ke gambar asli
        segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    else:
        # Jika tidak ada kontur, kembalikan gambar kosong
        segmented = np.zeros_like(img_rgb)

    return segmented, mask

# 5. Fungsi Ekstraksi Fitur (Tidak ada perubahan)
def extract_features(image, mask):
    features = {}
    if np.sum(mask) == 0: return None

    labeled_mask = label(mask)
    props_list = regionprops(labeled_mask)
    if not props_list: return None

    props = props_list[0]
    features['area'] = props.area
    features['aspect_ratio'] = props.major_axis_length / (props.minor_axis_length + 1e-6)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_pixels = hsv_image[mask > 0]
    features['hue_mean'] = np.mean(hsv_pixels[:, 0])
    features['saturation_mean'] = np.mean(hsv_pixels[:, 1])
    features['value_mean'] = np.mean(hsv_pixels[:, 2])

    return features

# 6. Fungsi Klasifikasi Berbasis Aturan (Tidak ada perubahan, aturan tetap relevan)
def classify_rice_by_rules_refined(features):
    if features['value_mean'] < 60:
        return "ketan_hitam"
    elif features['hue_mean'] < 25 and features['saturation_mean'] > 70:
        return "beras_merah"
    elif features['value_mean'] > 150 and features['saturation_mean'] < 50:
        if features['aspect_ratio'] > 2.5:
            return "IR64"
        elif features['aspect_ratio'] < 1.8:
            return "ketan_putih"
        else:
            return "pandan_wangi"
    else:
        return "pandan_wangi"

# 7. Loop Utama: Ekstraksi Fitur
features_csv_path = os.path.join(output_path, "rice_features_edge_detection.csv")
if not os.path.exists(features_csv_path):
    print("Memulai proses ekstraksi fitur menggunakan Edge Detection...")
    data = []
    for class_name in classes:
        class_dir = os.path.join(input_path, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Folder untuk kelas '{class_name}' tidak ditemukan.")
            continue

        print(f"Memproses kelas: {class_name}")
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            # Gunakan fungsi preprocessing edge detection yang baru
            segmented, mask = preprocess_with_edge_detection(img_path)

            if segmented is not None and mask is not None and np.sum(mask) > 0:
                output_img_path = os.path.join(output_path, "segmented_images_edge", f"{class_name}_{img_file}")
                cv2.imwrite(output_img_path, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))

                features = extract_features(segmented, mask)
                if features:
                    features['class'] = class_name
                    data.append(features)

    df = pd.DataFrame(data)
    df.to_csv(features_csv_path, index=False)
    print(f"Ekstraksi fitur selesai! File disimpan di: {features_csv_path}")
else:
    print(f"File fitur {features_csv_path} sudah ada. Memuat data...")
    df = pd.read_csv(features_csv_path)

# 8. Evaluasi Model Berbasis Aturan
def evaluate_rule_based_classification(df):
    print("\n--- Hasil Evaluasi Model (Preprocessing Edge Detection) ---")

    df['predicted_class'] = df.apply(classify_rice_by_rules_refined, axis=1)

    correct_predictions = (df['class'] == df['predicted_class']).sum()
    total_samples = len(df)
    accuracy = (correct_predictions / total_samples) * 100

    print(f"Akurasi Model: {accuracy:.2f}%\n")

    all_labels = sorted(classes)
    conf_matrix = pd.crosstab(df['class'], df['predicted_class'], rownames=['Aktual'], colnames=['Prediksi'])
    conf_matrix = conf_matrix.reindex(index=all_labels, columns=all_labels, fill_value=0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', xticklabels=all_labels, yticklabels=all_labels)
    plt.title('Confusion Matrix (Preprocessing Edge Detection)')
    plt.show()

# Panggil fungsi evaluasi
if 'df' in locals() and not df.empty:
    evaluate_rule_based_classification(df.copy())

# 9. Visualisasi untuk Penyesuaian Aturan (Tetap relevan)
def visualize_data_for_tuning(df):
    print("\n--- Visualisasi untuk Penyesuaian Aturan ---")

    df_white = df[df['class'].isin(['ketan_putih', 'IR64', 'pandan_wangi'])]
    plt.figure(figsize=(10, 6))
    sns.stripplot(data=df_white, x='class', y='aspect_ratio', order=['ketan_putih', 'pandan_wangi', 'IR64'])
    plt.title('Distribusi Aspect Ratio untuk Jenis Beras Putih')
    plt.ylabel('Aspect Ratio (Semakin tinggi semakin lonjong)')
    plt.xlabel('Kelas Beras')
    plt.grid(axis='y')
    plt.show()

# Panggil fungsi visualisasi
if 'df' in locals() and not df.empty:
    visualize_data_for_tuning(df)