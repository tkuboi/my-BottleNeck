import os
import urllib
import logging
import json

from logging import FileHandler
from flask import request
from flask import jsonify
from flask import Flask
from flask import Blueprint

from .net.recognizer import Recognizer

#graph = tf.get_default_graph()
recognizer = Recognizer() 
def create_app():
    flask = Flask(__name__)
    try:
        flask.config.from_pyfile(os.path.dirname(__file__) + '/config.py')
        print("importing configuration from config.py")
    except:
        print("config.py not found."+\
              "user environment variables or "+\
              "apache environmental variables will be used.")
    logging_path = os.path.dirname(__file__) + '/../logs/debug.log'
    file_handler = FileHandler(logging_path, 'a')
    file_handler.setLevel(logging.DEBUG)
    flask.logger.addHandler(file_handler)
    from .api_1_0 import api as api_1_0_blueprint
    flask.register_blueprint(api_1_0_blueprint, url_prefix='/api/v1.0')
    #from .main import main as main_blueprint
    #flask.register_blueprint(main_blueprint)
    recognizer.init_app(flask)
    recognizer.load_model()

    return flask
