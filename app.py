import os
from flask import Flask, render_template, request
import dbscan as model

app=Flask(__name__)
app.secret_key= os.urandom(30)

ALLOWED_EXTS = {"xlsx","csv"}
GLOBAL_FILE = "rand"

def check_file(file):
    return '.' in file and file.rsplit('.', 1)[1].lower() in ALLOWED_EXTS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    return render_template('image.html')


@app.route('/dbscan', methods=['post','get'])
def dbsc():
    if request.method =='POST':
        eps = request.form['eps']
        min_samples = request.form['min_samples']
        model.dbscan(float(eps), int(min_samples), GLOBAL_FILE)
    return render_template('image.html')


@app.route('/dashboard', methods=['post','get'])
def dashboard():
    error=None
    filename = None

    if request.method =='POST':
        if 'file' not in request.files:
            error="File not selected"
            return render_template('index.html', error=error)


        file = request.files['file']
        filename = file.filename
        global GLOBAL_FILE
        GLOBAL_FILE = f"uploads/{filename}"

        if filename =='':
            error = "Filename is empty"
            return render_template('index.html', error=error)

        if check_file(filename) == False:
            error = "This file is not allowed"
            return render_template('index.html', error=error)

        file.save(os.path.join("uploads", filename))

    return render_template('index.html', filename=filename)



if __name__ == "__main__":
    app.run(debug=True)