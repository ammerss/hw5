from flask import Flask, render_template, request , session
from train import *
from inference import *
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='./template')
app.secret_key = "super secret key"
UPLOAD_FOLDER = os.getcwd()
#UPLOAD_FOLDER = r"C:\Users\amyss\OneDrive\Documents\nyu-2\cloud_ml\hw5\src"


@app.route("/", methods=['GET', 'POST'])
@app.route('/index')
def index():
    if request.method == 'POST':
        if request.form.get('action1') == 'dry run train':
            print ("training")
            train_result = train()
            return render_template("index.html", train_result = train_result)
        
        elif  request.form.get('action2') == 'inference':

            # Upload file flask
            uploaded_img = request.files['uploaded-file']
            # Extracting uploaded data file name
            img_filename = uploaded_img.filename
            #img_filename = secure_filename(uploaded_img.filename)
            # Upload file to database (defined uploaded folder in static path)
            uploaded_img.save(os.path.join(UPLOAD_FOLDER, img_filename))
            # Storing uploaded file path in flask session
            session['uploaded_img_file_path'] = os.path.join(UPLOAD_FOLDER, img_filename)

            #print(session['uploaded_img_file_path'])
            print (uploaded_img)
            print(img_filename)
            inference_result = inference(UPLOAD_FOLDER + "\\" + img_filename)

            
            return render_template("index.html", inference_result = inference_result, user_image = os.path.join(UPLOAD_FOLDER, img_filename))
        
        
        else:
            pass # unknown
    elif request.method == 'GET':
        return render_template('index.html')
    
    return render_template("index.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
