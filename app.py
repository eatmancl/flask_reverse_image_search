from flask import Flask, request, jsonify,render_template
import matcher
import os
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import Pic_str as generator
import base64

app = Flask(__name__,static_url_path='/static')
CORS(app)
basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png','PNG', 'jpg', 'JPG'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/")
@cross_origin()
def home_view(): 
    return "<h1>server is running</h1>"

#upload and search
@app.route('/api/show', methods=['POST'])
@cross_origin()
def upload_show():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    print(basedir)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['image']
    if f and allowed_file(f.filename):
        fname = secure_filename(f.filename)
        # print (fname)
        ext = fname.rsplit('.', 1)[1]
        new_filename = generator.create_uuid() + '.' + ext
        new_filename = os.path.join(file_dir, new_filename)
        f.save(new_filename)
        # print('image has been uploaded successfully')
        # print(new_filename)
    else:
        return jsonify({"error": 1001, "msg": "fail"})
    # images = "static/dataset/"
    model = "static/features3.pck"
    return jsonify(matcher.run(new_filename,model,basedir))

@app.route('/upload_image',methods=['POST'])
@cross_origin()
def upload_image():
    # print('start')
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    # print(basedir)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['image']
    print(f)
    if f and allowed_file(f.filename):
        fname = secure_filename(f.filename)
        # print (fname)
        ext = fname.rsplit('.', 1)[1]
        new_filename = generator.create_uuid() + '.' + ext
        new_filename = os.path.join(file_dir, new_filename)
        f.save(new_filename)
        print({'code': 200, 'msg': 'image has been uploaded successfully'})
        return jsonify({'code': 200, 'msg': 'image has been uploaded successfully'})
    else:
        return jsonify({"error": 1001, "msg": "fail"})

@app.route('/test', methods=['GET'])
@cross_origin()
def post():
    return jsonify({'code': 200,'msg':'test'})

if __name__ == '__main__':
    app.run()
