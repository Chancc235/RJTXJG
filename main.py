from flask import Flask, request, render_template
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load the YOLO model
model = YOLO("yolo11n.pt")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploads', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        # Predict with the model
        results = model(filepath)
        result_image_path = os.path.join('static', 'result.jpg')
        results[0].save(filename=result_image_path)  # save to static folder

        return render_template('result.html', result_image='result.jpg')



if __name__ == '__main__':
    app.run(debug=True)
    print('Server starting...')