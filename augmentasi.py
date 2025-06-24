#Import library yang diperlukan
import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm 

# Membaca path ke direktori saat ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

datagen = ImageDataGenerator(
    rescale=1./255,          # Normalisasi pixel [0,1]
    validation_split=0.2     # Pisahkan 20% data untuk validasi
)

train_generator = datagen.flow_from_directory(
    '/content/drive/MyDrive/DATASET DIP/',
    target_size=(224, 224),  # Resize semua gambar ke 224x224
    batch_size=32,
    class_mode='categorical',
    subset='training'        # 80% untuk training
)

val_generator = datagen.flow_from_directory(
    '/content/drive/MyDrive/DATASET DIP/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'      # 20% untuk validasi
)

#Augementasi gambar jika dataset kecil (<1000 gambar per kelas)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,       # Rotasi acak 0-30 derajat
    zoom_range=0.2,          # Zoom acak 80-120%
    width_shift_range=0.1,   # Geser horizontal acak
    height_shift_range=0.1,  # Geser vertikal acak
    horizontal_flip=True,    # Flip horizontal acak
    validation_split=0.2
)

# Memastikan keseimbangan kelas 
for class_name in ['IR64', 'Ketan_Hitam', 'Ketan_Putih', 'Merah', 'Pandan_Wangi']:
    print(f"{class_name}: {len(os.listdir(f'/content/drive/MyDrive/DATASET DIP/{class_name}'))} gambar")
    
# Memotong gambar yang terlalu besar

# Konfigurasi
TARGET_SIZE = (224, 224)
# Path folder dataset
dataset_path = "/content/drive/MyDrive/Dataset_Beras/training"
output_path = "/content/drive/MyDrive/Dataset_Beras/processed/"

# Buat folder output
os.makedirs(output_path, exist_ok=True)

# Loop dengan penanganan error
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    output_class_path = os.path.join(output_path, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    for img_name in tqdm(os.listdir(class_path)):
        img_path = os.path.join(class_path, img_name)

        try:
            # Baca gambar dengan cv2 + cek keberadaan
            img = cv2.imread(img_path)
            if img is None:
                print(f"Gagal baca: {img_path}")
                continue  # Lewati gambar error

            # Resize dan simpan
            img_resized = cv2.resize(img, TARGET_SIZE)
            cv2.imwrite(os.path.join(output_class_path, img_name), img_resized)

        except Exception as e:
            print(f"Error pada {img_path}: {str(e)}")
            
            
# Menajamkan gambar yang sudah dipotong

# Path folder dataset
dataset_path =  "/content/drive/MyDrive/DATASET DIP/DATASET_RESIZED/"
output_path = "/content/drive/MyDrive/DATASET DIP/DATASET_SHARPEN/"

# Buat folder output
os.makedirs(output_path, exist_ok=True)

# Loop dengan penanganan error
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    output_class_path = os.path.join(output_path, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    for img_name in tqdm(os.listdir(class_path)):
        img_path = os.path.join(class_path, img_name)

        try:
            # Baca gambar dengan cv2 + cek keberadaan
            img = cv2.imread(img_path)
            if img is None:
                print(f"Gagal baca: {img_path}")
                continue  # Lewati gambar error

            # 2. Proses sharpening dengan Unsharp Masking
            blurred = cv2.GaussianBlur(img, (0, 0), 3)  # Perbaikan: gunakan img bukan dataset_path
            sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

            # Simpan gambar yang sudah di-sharpen
            output_img_path = os.path.join(output_class_path, img_name)
            cv2.imwrite(output_img_path, sharpened)

        except Exception as e:
            print(f"Error pada {img_path}: {str(e)}")