from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
import time

app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model_path = "./models/bert-base-chinese-local"
distilled_model_path = "./models/distilled_bilstm.pth"

tokenizer = BertTokenizer.from_pretrained(model_path)


class BiLSTMStudent(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256, num_layers=2, num_labels=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=tokenizer.pad_token_id)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeds)
        lengths = attention_mask.sum(dim=1) - 1
        pooled = lstm_out[torch.arange(lstm_out.size(0)), lengths]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


vocab_size = len(tokenizer)
model = BiLSTMStudent(vocab_size=vocab_size)
model.load_state_dict(torch.load(distilled_model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()
print("蒸馏模型加载完成！")


def predict(text):
    start_time = time.time()
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()

    inference_time = (time.time() - start_time) * 1000

    return {
        'text': text,
        'prediction': '正面' if predicted_class == 1 else '负面',
        'confidence': {
            '负面': round(probabilities[0][0].item(), 4),
            '正面': round(probabilities[0][1].item(), 4)
        },
        'inference_time_ms': round(inference_time, 2)
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': '请提供文本内容'}), 400
        
        result = predict(text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch_predict', methods=['POST'])
def api_batch_predict():
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'error': '请提供文本列表'}), 400
        
        results = []
        for text in texts:
            result = predict(text)
            results.append(result)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


import os

@app.route('/api/model_info', methods=['GET'])
def api_model_info():
    total_params = sum(p.numel() for p in model.parameters())
    model_file_size = os.path.getsize(distilled_model_path) / (1024 * 1024)  # 计算 MB
    return jsonify({
        'model_name': '蒸馏BiLSTM情感分析模型',
        'architecture': '双层双向LSTM',
        'vocab_size': vocab_size,
        'parameters': f'{total_params / 1e6:.2f}M',
        'model_size_mb': round(model_file_size, 2),  # 新增
        'max_sequence_length': 128,
        'device': str(device),
        'classes': ['负面', '正面']
    })


if __name__ == '__main__':
    print("服务启动中...")
    print("访问地址: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)