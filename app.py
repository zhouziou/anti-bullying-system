import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import threading
from queue import Queue
from datetime import timedelta

# 初始化Flask应用
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///reports.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'mov', 'avi', 'mkv'}
# 后端实现
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB限制
app.config['MODEL_PATH'] = 'model/best_model.pth'
app.secret_key = 'your-secret-key-123'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
db = SQLAlchemy(app)


# ======================
# 数据库模型
# ======================
class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_filename = db.Column(db.String(200))
    text = db.Column(db.String(500))
    phone = db.Column(db.String(20))
    prediction = db.Column(db.String(50), default='待分析')
    confidence = db.Column(db.Float, default=0.0)
    details = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, server_default=db.func.now())


class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(100))


# ======================
# 新版暴力检测模型（与训练保持一致）
# ======================
class C3DViolenceDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(3, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)), nn.Dropout3d(0.1),
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.MaxPool3d(2), nn.Dropout3d(0.15),
            nn.Conv3d(128, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU(),
            nn.Conv3d(256, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU(),
            nn.MaxPool3d(2), nn.Dropout3d(0.2),
            nn.Conv3d(256, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU(),
            nn.Conv3d(512, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU(),
            nn.MaxPool3d(2), nn.Dropout3d(0.25),
            nn.Conv3d(512, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU(),
            nn.Conv3d(512, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU(),
            nn.MaxPool3d(2), nn.Dropout3d(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 1 * 3 * 3, 1024), nn.ReLU(), nn.Dropout(0.8),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.7),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(1)


# ======================
# 视频预处理与检测器
# ======================
class ViolenceDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = C3DViolenceDetector().to(self.device)
        # 加载checkpoint字典
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()
        # 配置参数：视频需要16帧、大小为112x112
        self.clip_length = 16
        self.frame_size = (112, 112)

    # ===== 以下为预处理的调整部分 =====
    def temporal_sampling(self, frames):
        """均匀采样16帧"""
        total = len(frames)
        if total >= self.clip_length:
            indices = np.linspace(0, total - 1, self.clip_length, dtype=int).tolist()
        else:
            indices = list(range(total))
            while len(indices) < self.clip_length:
                indices.append(total - 1)
        return [frames[i] for i in indices]

    def center_crop(self, frame):
        """中心裁剪到目标尺寸"""
        h, w = frame.shape[:2]
        th, tw = self.frame_size
        if h < th or w < tw:
            scale = max(th / h, tw / w)
            new_size = (int(w * scale), int(h * scale))
            frame = cv2.resize(frame, new_size)
            h, w = frame.shape[:2]
        top = (h - th) // 2
        left = (w - tw) // 2
        return frame[top:top + th, left:left + tw]

    def preprocess_video(self, video_path):
        # 读取视频帧
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if not frames:
            # 返回全0张量
            return torch.zeros((1, 3, self.clip_length, *self.frame_size)).to(self.device)

        # 均匀采样16帧
        sampled_frames = self.temporal_sampling(frames)
        processed = []
        for f in sampled_frames:
            # 中心裁剪
            f = self.center_crop(f)
            # 再resize确保与目标尺寸匹配（如果center_crop后尺寸已经正确可省略此步）
            f = cv2.resize(f, self.frame_size)
            processed.append(f)
        arr = np.stack(processed, axis=0).astype(np.float32) / 255.0  # (T, H, W, C)
        arr = np.transpose(arr, (3, 0, 1, 2))  # 转为 (C, T, H, W)
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    # ===== 预处理调整结束 =====

    def predict(self, video_path):
        with torch.no_grad():
            inputs = self.preprocess_video(video_path)  # 预处理
            outputs = self.model(inputs)  # 前向传播
            prob = torch.sigmoid(outputs).cpu().numpy()[0]  # 计算概率
            prediction = "暴力行为" if prob >= 0.5 else "非暴力行为"
            confidence = float(prob) if prob >= 0.5 else float(1 - prob)
            details = {
                "非暴力行为概率": float(1 - prob),
                "暴力行为概率": float(prob)
            }
            return prediction, confidence, details


# 初始化检测器
try:
    detector = ViolenceDetector(app.config['MODEL_PATH'])
    print("暴力检测模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    detector = None

analysis_queue = Queue()


def background_analyzer():
    while True:
        # 获取队列任务
        report_id, video_path = analysis_queue.get()
        try:
            if detector:
                # 调用模型分析
                prediction, confidence, details = detector.predict(video_path)
                # 更新数据库信息
                report = Report.query.get(report_id)
                if report:
                    report.prediction = prediction
                    report.confidence = confidence
                    report.details = str(details)
                    db.session.commit()
                    print(f"报告 {report_id} 分析完成: {prediction} (置信度: {confidence:.2f})")
        except Exception as e:
            print(f"分析报告 {report_id} 时出错: {str(e)}")
        finally:
            analysis_queue.task_done()


# 启动后台线程
if detector:
    analyzer_thread = threading.Thread(target=background_analyzer, daemon=True)
    analyzer_thread.start()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('report.html')


@app.route('/report', methods=['POST'])
def handle_report():
    if 'video' not in request.files:
        flash('请上传视频文件！', 'error')
        return redirect(url_for('index'))

    video_file = request.files['video']
    text = request.form.get('text', '').strip()
    phone = request.form.get('phone', '').strip()

    if video_file.filename == '':
        flash('请选择视频文件！', 'error')
        return redirect(url_for('index'))

    if not text:
        flash('事件描述不能为空！', 'error')
        return redirect(url_for('index'))

    if not phone or not phone.isdigit() or len(phone) != 11:
        flash('请输入有效的11位手机号码！', 'error')
        return redirect(url_for('index'))

    if not allowed_file(video_file.filename):
        flash('只支持MP4、MOV、AVI、MKV格式的视频文件！', 'error')
        return redirect(url_for('index'))

    filename = secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{video_file.filename}")
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(video_path)

    # 写入数据库，初始状态设置为待分析
    new_report = Report(
        video_filename=filename,
        text=text,
        phone=phone,
        prediction="待分析",
        confidence=0.0,
        details="等待模型分析"
    )
    db.session.add(new_report)
    db.session.commit()

    if detector:
        try:
            # 直接调用预测函数进行自动分析
            prediction, confidence, details = detector.predict(video_path)
            new_report.prediction = prediction
            new_report.confidence = confidence
            new_report.details = str(details)
            db.session.commit()
            flash('报告提交成功，视频已自动分析！', 'success')
        except Exception as e:
            flash(f'报告提交成功，但视频自动分析失败：{str(e)}', 'error')
    else:
        flash('报告提交成功！(模型未加载，请管理员手动分析)', 'warning')

    return redirect(url_for('index'))


@app.route('/uploads/<filename>')
# 后端实现
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/admin_login', methods=['POST'])
def admin_login():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()

    admin = Admin.query.filter_by(username=username).first()
    if admin and check_password_hash(admin.password, password):
        session['admin_logged_in'] = True  # 设置登录状态
        session['admin_username'] = username
        return redirect(url_for('admin_dashboard'))
    else:
        flash('管理员账号或密码错误！', 'error')
        return redirect(url_for('index'))


@app.route('/admin')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        flash('请先登录管理员账号！', 'error')
        return redirect(url_for('index'))
    reports = Report.query.order_by(Report.id.desc()).all()
    # 转换创建时间为中国标准时间 (UTC+8)
    for report in reports:
        report.local_created_at = report.created_at + timedelta(hours=8)
    return render_template('admin.html',
                           reports=reports,
                           admin_username=session.get('admin_username'),
                           detector_loaded=detector is not None)


@app.route('/update_prediction/<int:report_id>', methods=['POST'])
# 后端实现
def update_prediction(report_id):
    # 权限验证
    if not session.get('admin_logged_in'):
        return jsonify({'status': 'error', 'message': '未授权'}), 401
    # 获取请求
    report = Report.query.get_or_404(report_id)
    prediction = request.json.get('prediction')
    if prediction not in ['暴力行为', '非暴力行为']:
        return jsonify({'status': 'error', 'message': '无效的预测结果'}), 400
    # 更新数据库
    report.prediction = prediction
    report.details = f"管理员手动标记为{prediction}"
    db.session.commit()
    return jsonify({
        'status': 'success',
        'message': '预测结果已更新',
        'prediction': prediction
    })


@app.route('/analyze_now/<int:report_id>', methods=['POST'])
def analyze_now(report_id):
    if not session.get('admin_logged_in'):
        return jsonify({'status': 'error', 'message': '未授权访问'}), 401

    if not detector:
        return jsonify({'status': 'error', 'message': '模型未加载'}), 400

    report = Report.query.get_or_404(report_id)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], report.video_filename)
    try:
        prediction, confidence, details = detector.predict(video_path)
        report.prediction = prediction
        report.confidence = confidence
        report.details = str(details)
        db.session.commit()
        return jsonify({
            'status': 'success',
            'message': '分析完成',
            'prediction': prediction,
            'confidence': confidence,
            'details': details
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'分析失败: {str(e)}'}), 500


@app.route('/admin_logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    session.pop('admin_username', None)
    flash('您已成功退出管理员账号', 'info')
    return redirect(url_for('index'))


with app.app_context():
    db.create_all()
    if not Admin.query.filter_by(username="admin").first():
        hashed_pw = generate_password_hash("admin123")
        db.session.add(Admin(username="admin", password=hashed_pw))
        db.session.commit()

if __name__ == '__main__':
    app.run(debug=True)
