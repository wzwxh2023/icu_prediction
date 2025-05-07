from http.server import BaseHTTPRequestHandler
import json
import os
import joblib
import numpy as np
import pandas as pd

# 模型路径
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../model')
MODEL_PATH = os.path.join(MODEL_DIR, 'lightgbm_icu_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, 'feature_list.txt')

# 加载模型和相关文件(只会在冷启动时执行一次)
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    with open(FEATURE_LIST_PATH, 'r') as f:
        feature_list = [line.strip() for line in f.readlines()]
    
    print("模型和相关文件加载成功!")
except Exception as e:
    print(f"加载模型出错: {str(e)}")
    model = None
    scaler = None
    feature_list = []

def process_features(data):
    """从请求数据提取和处理特征"""
    # 提取生理指标
    raw_physiological = ['age', 'bmi', 'pulse', 'tempreture', 'sbp', 'res']
    mews_total = ['mews_total']
    
    # 提取分类特征
    categorical_features = [
        'gender', 'admission_unit', 'surgey', 'intervention',
        'exam_critical_flag', 'lab_critical_flag',
        'o2', 'mews_aware'
    ]
    
    # 合并数值特征
    numeric_features = raw_physiological + mews_total
    
    # 处理数值特征
    numeric_data = {}
    for feature in numeric_features:
        numeric_data[feature] = data.get(feature, 0)  # 如果缺失则默认为0
    
    # 应用标准化
    if scaler:
        numeric_df = pd.DataFrame([numeric_data])
        numeric_scaled = scaler.transform(numeric_df)
        numeric_df = pd.DataFrame(numeric_scaled, columns=numeric_features)
    else:
        numeric_df = pd.DataFrame([numeric_data])
    
    # 处理分类特征
    categorical_data = {}
    for feature in categorical_features:
        categorical_data[feature] = data.get(feature, '')
        
    categorical_df = pd.DataFrame([categorical_data])
    categorical_df = pd.get_dummies(categorical_df, drop_first=True)
    
    # 确保所有需要的分类特征列都存在
    for feature in feature_list:
        if feature not in numeric_df.columns and feature not in categorical_df.columns:
            if not feature.startswith('diagnosis_emb_') and not feature.startswith('history_emb_') and \
               not feature.startswith('exam_critical_value_emb_') and not feature.startswith('lab_critical_value_emb_'):
                categorical_df[feature] = 0
    
    # 处理文本嵌入特征
    if 'embeddings' in data:
        embeddings = data['embeddings']
        for emb_name, emb_values in embeddings.items():
            for i, val in enumerate(emb_values):
                column_name = f"{emb_name}_emb_{i}"
                if column_name in feature_list:
                    numeric_df[column_name] = val
                elif i < 768:  # 假设嵌入向量维度为768，适用于BAAI/bge-large模型
                    # 如果特征列表中没有该特征，但是在预期的嵌入维度范围内，仍然添加
                    numeric_df[column_name] = val
    
    # 合并所有特征
    all_features = pd.concat([numeric_df.reset_index(drop=True), 
                             categorical_df.reset_index(drop=True)], 
                             axis=1)
    
    # 确保特征顺序与训练时一致
    final_features = pd.DataFrame(columns=feature_list)
    for col in feature_list:
        if col in all_features.columns:
            final_features[col] = all_features[col]
        else:
            final_features[col] = 0
    
    return final_features

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """处理OPTIONS请求（用于CORS预检）"""
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')  # 在生产环境中应限制为您的前端域名
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
    def do_POST(self):
        """处理POST请求"""
        # 获取请求内容长度
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        # 解析JSON数据
        try:
            data = json.loads(post_data.decode('utf-8'))
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': '无效的JSON数据'}).encode())
            return
        
        # 检查是否提供了嵌入向量
        if 'embeddings' not in data:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {
                'error': '请提供嵌入向量。直接文本处理在此部署中不支持。'
            }
            self.wfile.write(json.dumps(response).encode())
            return
            
        # 验证模型是否已加载
        if model is None:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': '模型未加载成功'}).encode())
            return
        
        try:
            # 处理特征
            features = process_features(data)
            
            # 预测
            prediction_proba = model.predict_proba(features)[0][1]
            prediction = int(prediction_proba > 0.5)  # 使用0.5作为默认阈值
            
            # 返回结果
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                'icu_probability': float(prediction_proba),
                'icu_needed': bool(prediction == 1),
                'prediction_timestamp': pd.Timestamp.now().isoformat()
            }
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': f'预测错误: {str(e)}'}).encode())
