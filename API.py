from flask import Flask, request, render_template, jsonify
import os
app = Flask(__name__)
@app.route('/')
def index():
return render_template('index.html')
@app.route('/generate', methods=['POST'])
def generate():
data = request.get_json()
prompt = data['prompt']
image_url = "static/images/generated_image.jpg" 
return jsonify({'image_url': image_url})
if __name__ == '__main__':
app.run(debug=True)
