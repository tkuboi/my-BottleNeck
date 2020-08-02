import os

from flask import render_template, redirect, url_for, abort, flash, request,\
    current_app, make_response, send_from_directory
from . import main
from .. import recognizer
from ..common.utils import generate_filename

@main.route('/', methods=['GET', 'POST'])
def index():
    return 'Hello World!'

@main.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        fp = request.files['file']
        if fp.filename == '':
            flash('No selected file')
            return redirect(request.url)
        fp = request.files['file']
        imagefile = generate_filename(fp.filename)
        imagefile_path = os.path.join(current_app.config['UPLOAD_FOLDER'], imagefile)
        fp.save(imagefile_path)
        return redirect(url_for('.uploaded_file', filename=imagefile))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@main.route('/uploaded_file/<filename>', methods=['GET'])
def uploaded_file(filename):
    imagefile_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    wines, box = recognizer.predict(imagefile_path)
    return render_template('result.html', filename=filename, wines=wines, box=box)
    #return redirect(url_for('static/uploaded', filename=filename))
