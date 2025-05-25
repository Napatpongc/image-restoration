# orchestrator/app.py

import os
import sys
from flask import Flask, render_template, request, send_from_directory

# ─── PATH SETUP ─────────────────────────────────────────────────────
BASE_DIR     = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)  # เพื่อให้ import modules.arf.run_arf ได้

TEMPLATE_DIR  = os.path.join(PROJECT_ROOT, 'templates')
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'static', 'uploads')

# ─── IMPORT APPLY_ARF ───────────────────────────────────────────────
from modules.arf.run_arf import apply_arf

# ─── FLASK APP ──────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=None
)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ─── serve style.css ────────────────────────────────────────────────
@app.route('/style.css')
def style_css():
    return send_from_directory(TEMPLATE_DIR, 'style.css')

# ─── serve spinner.svg ──────────────────────────────────────────────
@app.route('/spinner.svg')
def spinner_svg():
    return send_from_directory(
        os.path.join(PROJECT_ROOT, 'static'),
        'spinner.svg',
        mimetype='image/svg+xml'
    )

# ─── serve uploaded images ──────────────────────────────────────────
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ─── INDEX PAGE ─────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# ─── PROCESS PIPELINE ───────────────────────────────────────────────
@app.route('/process', methods=['POST'])
def process_image():
    # 1) รับไฟล์ภาพจากฟอร์ม
    file = request.files.get('image')
    if not file:
        return "กรุณาเลือกไฟล์ภาพ", 400

    # 2) บันทึกไฟล์ต้นทาง
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filename = file.filename
    inp_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(inp_path)

    # 3) รัน DEM → โมเดลทั้งหมด (apply_arf) แล้วได้ path ผลลัพธ์
    out_path = apply_arf(inp_path)
    out_name = os.path.basename(out_path)

    # 4) แสดงผลลัพธ์บน result.html
    return render_template(
        'result.html',
        filename=out_name,
        message='',     # ใช้ข้อความ default บน result.html
        kind=None,
        sigma=None,
        processed=True
    )

# ─── DOWNLOAD ENDPOINT ─────────────────────────────────────────────
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

# ─── MAIN ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
