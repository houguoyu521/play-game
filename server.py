from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from matminer.featurizers.composition import ElementProperty, Stoichiometry
from matminer.featurizers.conversions import StrToComposition
import uuid

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
user_sessions = {}  # 存储用户会话数据

@app.errorhandler(Exception)
def handle_error(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    return jsonify({
        "error": str(e),
        "code": code
    }), code

def clean_data(df, composition_col):
    df_cleaned = df.copy()
    
    if composition_col not in df.columns:
        raise ValueError(f"数据中未找到用户选择的成分列：'{composition_col}'")
    
    df_cleaned[composition_col] = df_cleaned[composition_col].astype(str)
    df_cleaned = df_cleaned.dropna(subset=[composition_col])
    
    for col in df_cleaned.columns:
        if col == composition_col:
            continue
        if df_cleaned[col].dtype == object:
            if df_cleaned[col].isnull().any():
                df_cleaned = df_cleaned.drop(columns=[col])
        else:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
    
    df_cleaned = df_cleaned.dropna(how='any')
    return df_cleaned

def transform_features_core(df, composition_col):
    transformation_status = {}
    
    try:
        df = StrToComposition().featurize_dataframe(df, composition_col, ignore_errors=True)
        if 'composition' not in df.columns:
            raise ValueError("无法将成分列转换为composition对象，请检查化学式格式（如Fe2O3）")
    except Exception as e:
        raise ValueError(f"成分转换失败: {str(e)}")

    original_numerical = df.select_dtypes(include=[np.number]).drop(columns=['composition'], errors='ignore')
    transformation_status["OriginalNumerical"] = f"保留原始数值列：{list(original_numerical.columns)}" if not original_numerical.empty else "无原始数值列"

    try:
        ep_featurizer = ElementProperty.from_preset("magpie")
        df_ep = ep_featurizer.featurize_dataframe(df, 'composition', ignore_errors=True)
        ep_features = [col for col in ep_featurizer.feature_labels() if col in df_ep.columns]
        transformation_status["ElementProperty"] = f"完成（生成{len(ep_features)}个特征）" if ep_features else "失败（无有效特征）"
    except Exception as e:
        raise ValueError(f"元素属性特征生成失败: {str(e)}")

    try:
        st_featurizer = Stoichiometry()
        df_st = st_featurizer.featurize_dataframe(df, 'composition', ignore_errors=True)
        st_features = [col for col in st_featurizer.feature_labels() if col in df_st.columns]
        transformation_status["Stoichiometry"] = f"完成（生成{len(st_features)}个特征）" if st_features else "失败（无有效特征）"
    except Exception as e:
        raise ValueError(f"化学计量特征生成失败: {str(e)}")

    df_processed = pd.concat([original_numerical, df_ep, df_st], axis=1)
    numerical_df = df_processed.select_dtypes(include=[np.number])

    if numerical_df.isnull().any().any():
        null_cols = numerical_df.columns[numerical_df.isnull().any()].tolist()
        numerical_df = numerical_df.fillna(numerical_df.mean())
        transformation_status["NullHandling"] = f"处理空值特征：{len(null_cols)}个特征（{null_cols}）已填充均值"
    else:
        transformation_status["NullHandling"] = "无空值特征"

    if numerical_df.empty:
        raise ValueError("未生成有效数值特征")
    
    return numerical_df, transformation_status, len(numerical_df.columns), ep_features, st_features, df_processed

@app.route('/')
def index():
    return render_template('client.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']
        if not file:
            return jsonify({"error": "未上传文件"}), 400
        
        user_id = str(uuid.uuid4())
        df = pd.read_excel(file) if file.filename.endswith(('.xls', '.xlsx')) else pd.read_csv(file)
        user_sessions[user_id] = {
            "raw": df.to_dict(orient='list'),
            "columns": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": df.head(5).replace({np.nan: None}).to_dict(orient='list')
        }
        
        return jsonify({
            "user_id": user_id,
            "columns": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": df.head(5).replace({np.nan: None}).to_dict(orient='list')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/transform_features', methods=['POST'])
def transform_features():
    try:
        req_data = request.get_json()
        user_id = req_data.get('user_id')
        composition_col = req_data.get('composition_col')
        
        if not user_id or user_id not in user_sessions:
            return jsonify({"error": "无效的用户会话，请重新上传数据"}), 400
        if not composition_col:
            return jsonify({"error": "未选择成分列"}), 400

        user_session = user_sessions[user_id]
        if not user_session['raw']:
            return jsonify({"error": "No data uploaded"}), 400

        df = pd.DataFrame(user_session['raw'])
        df_cleaned = clean_data(df, composition_col)

        df_processed, status, valid_feature_count, ep_features, st_features, full_df_processed = transform_features_core(df_cleaned, composition_col)

        transformed_preview = full_df_processed.drop(columns=['composition'], errors='ignore').head(5).replace({np.nan: None}).to_dict(orient='list')
        original_columns = list(df_processed.select_dtypes(include=[np.number]).drop(columns=['composition'], errors='ignore').columns)
        generated_features = ep_features + st_features

        user_session['transformed'] = df_processed.to_dict(orient='list')
        user_session['cleaned_transformed'] = df_cleaned.loc[df_processed.index].to_dict(orient='list')
        
        return jsonify({
            "transformation_status": status,
            "raw_feature_count": len(ep_features) + len(st_features) + len(original_columns),
            "valid_feature_count": valid_feature_count,
            "sample_count": len(df_processed),
            "transformed_preview": transformed_preview,
            "original_columns": original_columns,
            "generated_features": generated_features
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/train', methods=['POST'])
def start_training():
    try:
        req_data = request.get_json()
        user_id = req_data.get('user_id')
        if not user_id or user_id not in user_sessions:
            return jsonify({"error": "无效的用户会话，请重新上传数据"}), 400

        user_session = user_sessions[user_id]
        if not user_session.get('transformed') or not user_session.get('cleaned_transformed'):
            return jsonify({"error": "请先完成特征转化"}), 400

        df_processed = pd.DataFrame(user_session['transformed'])
        df_cleaned_transformed = pd.DataFrame(user_session['cleaned_transformed'])
        target_col = req_data['target_col']

        if target_col not in df_cleaned_transformed.columns:
            raise ValueError(f"目标列 {target_col} 不存在于原始数据中")

        df_processed[target_col] = df_cleaned_transformed[target_col]

        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]

        if y.isnull().any():
            raise ValueError(f"目标列 '{target_col}' 存在空值，请检查目标列数据")
        null_cols = X.columns[X.isnull().any()].tolist()
        if null_cols:
            raise ValueError(f"特征数据中以下列存在空值: {null_cols}，请检查数据质量或特征生成逻辑")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_mapping = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(),
            'gradient_boosting': GradientBoostingRegressor(),
            'extra_trees': ExtraTreesRegressor()
        }
        model = model_mapping[req_data['model']]
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

        return jsonify({
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "y_train": y_train.tolist(),
            "y_pred_train": y_pred_train.tolist(),
            "y_test": y_test.tolist(),
            "y_pred_test": y_pred_test.tolist(),
            "train_indices": X_train.index.tolist(),
            "test_indices": X_test.index.tolist(),
            "sample_stats": {
                "initial": len(pd.DataFrame(user_session['raw'])),
                "after_cleaning": len(clean_data(pd.DataFrame(user_session['raw']), req_data['composition_col'])),
                "after_feature_gen": len(df_processed),
                "test_set": len(y_test),
                "train_set": len(y_train)
            },
            "feature_count": X.shape[1],
            "feature_status": "所有特征值已成功生成"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 新增：下载特征表格接口
@app.route('/download_features', methods=['POST'])
def download_features():
    try:
        req_data = request.get_json()
        user_id = req_data.get('user_id')
        if not user_id or user_id not in user_sessions:
            return jsonify({"error": "无效的用户会话，请重新上传数据"}), 400
        
        user_session = user_sessions[user_id]
        if not user_session.get('transformed'):
            return jsonify({"error": "请先完成特征转化"}), 400
        
        df = pd.DataFrame(user_session['transformed'])
        csv_data = df.to_csv(index=False)
        
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=features.csv"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002)