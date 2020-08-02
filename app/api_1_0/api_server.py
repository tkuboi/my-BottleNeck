"""File for API Service
"""

import os
import sys
import re
import traceback
import logging
import json
import multiprocessing as mp
import subprocess

from . import api
from .. import recognizer
from ..common.utils import generate_filename
from flask import request
from flask import jsonify
from flask import Flask
from flask import Blueprint
from flask import current_app

@api.route('/hello/<name>', methods=['GET'])
def hello_world(name):
    #recognizer.load_model()
    status_code = 200
    reply = {"message":"Hello %s" % (name)}
    return jsonify(reply),status_code

@api.route('/get_wine/<imagefile>', methods=['GET'])
def get_wine(imagefile):
    imagefile_path = os.path.join(current_app.config['UPLOAD_FOLDER'], imagefile)
    wines, box = recognizer.predict(imagefile_path)
    status_code = 200
    reply = {'wines': wines, 'box': box}
    return jsonify(reply),status_code

@api.route('/post_wine', methods=['POST'])
def post_wine():
    fp = request.files['data']
    if fp.filename == '':
        reply = {'message': 'No selected file'}
        status_code = 200
        return jsonify(reply), status_code
    imagefile = generate_filename(fp.filename)
    imagefile_path = os.path.join(current_app.config['UPLOAD_FOLDER'], imagefile)
    fp.save(imagefile_path)
    wines, box = recognizer.predict(imagefile_path)
    reply = {'wines': wines, 'box': box}
    status_code = 201
    return jsonify(reply), status_code
