from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import tempfile
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
CORS(app)

user_data = {}

def clean_data(df):
    """数据清洗：删除含NaN/NaNo的行"""
    df = df.dropna(how='any')
    for col in df.columns:
        if df[col].dtype == object:
            df = df[~df[col].str.contains('nano', case=False, na=False)]
    return df.reset_index(drop=True)

def process_upload(file_path, ext):
    """处理上传文件（根据扩展名解析）"""
    if ext == 'csv':
        df = pd.read_csv(file_path)
    elif ext in {'xls', 'xlsx'}:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("不支持的文件格式，仅支持csv、xls、xlsx")
    df = clean_data(df)
    return df.to_dict(orient='list')

def generate_features(df, composition_col):
    """生成特征（含错误处理）"""
    def safe_composition(x):
        try:
            return Composition(x).reduced_formula
        except:
            return None
    
    df['composition'] = df[composition_col].apply(safe_composition)
    df = df.dropna(subset=['composition'])  # 过滤无效组成
    
    featurizer = ElementProperty.from_preset("magpie")
    df = featurizer.featurize_dataframe(df, 'composition', ignore_errors=True)
    return df.select_dtypes(include=[np.number])

@app.route('/')
def index():
    return render_template('client.html')

@app.route('/upload', methods=['POST'])
def handle_upload():
    try:
        file = request.files['file']
        if not file:
            raise ValueError("未选择文件")
        
        ext = file.filename.split('.')[-1].lower()
        if ext not in {'csv', 'xls', 'xlsx'}:
            raise ValueError("不支持的文件格式，仅支持csv、xls、xlsx")
        
        _, temp_path = tempfile.mkstemp(suffix=f".{ext}")
        file.save(temp_path)
        data_dict = process_upload(temp_path, ext)
        os.remove(temp_path)
        
        user_data['raw'] = data_dict  # 存储原始数据
        return jsonify({
            "columns": list(data_dict.keys()),
            "preview": {k: v[:5] for k, v in data_dict.items()}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # 增加数据存在性检查
        if 'raw' not in user_data:
            raise ValueError("请先上传数据文件")
            
        req_data = request.get_json()
        target_col = req_data['target_col']
        composition_col = req_data['composition_col']
        
        df = pd.DataFrame(user_data['raw'])
        if target_col not in df.columns or composition_col not in df.columns:
            raise ValueError("选择的列不存在")
        
        df_processed = generate_features(df, composition_col)
        df_processed[target_col] = df[target_col]
        
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 修复：旧版sklearn不支持'squared'参数，手动计算RMSE
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        return jsonify({
            "rmse": rmse,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "features": X.columns.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, debug=True)