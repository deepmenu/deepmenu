from __future__ import absolute_import

from flask import Flask, render_template, url_for, request,jsonify
import pandas as pd

# Your workspace need to be in PYTHONPATH
# In VSCode: add '"env": {"PYTHONPATH" : "${workspaceFolder}"}' to your launch.json
from core.src.pipeline import Pipeline

pipeline = Pipeline()
app = Flask(__name__)

@app.route('/pick', methods=['GET', 'POST'])
def pick():
    id = int(float(request.get_data()))
      
    print(id)
    items = pipeline.predict(id)
    
    return jsonify([item.to_dict() for item in items])

@app.route('/')
@app.route('/home')
def home():
    
    items = pipeline.get_items()

    return render_template('home.html', items=items, length=len(items), col_num=3)


if __name__ == "__main__":
    
    app.run(debug=True)