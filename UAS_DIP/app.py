from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error="Tidak ada file yang dipilih")
    
    file = request.files['image']
    
    if file.filename == '':
        return render_template('index.html', error="Tidak ada file yang dipilih")
        
    if not allowed_file(file.filename):
        return render_template('index.html', error="Format file tidak didukung. Gunakan JPG, PNG, atau GIF")
        
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # ============================================
        # ADD YOUR CLASSIFICATION LOGIC HERE
        # Replace this example with your actual model
        # ============================================
        prediction = "5.2 kg"  # Example prediction
        
        return render_template('index.html', 
                             prediction=prediction,
                             filename=filename)
    
    except Exception as e:
        return render_template('index.html', error=f"Terjadi error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)