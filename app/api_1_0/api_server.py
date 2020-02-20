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
    label, score = recognizer.predict(imagefile)
    status_code = 200
    reply = {"label": label, "score": str(round(score, 2))}
    return jsonify(reply),status_code
