from flask import Flask, render_template, request
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)

        # Di sini nanti kamu bisa klasifikasikan gambar dengan model ML
        result = "Contoh Kelas: Kucing"

        return render_template("index.html", result=result, image_path=image_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
