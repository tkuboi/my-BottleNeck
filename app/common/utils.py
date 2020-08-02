import os
import datetime
import hashlib

def generate_filename(original):
    name, ext = os.path.splitext(original)
    m = hashlib.md5()
    strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    m.update('salt'.encode('utf-8'))
    m.update(strtime.encode('utf-8'))
    m.update(name.encode('utf-8'))
    return m.hexdigest() + ext
