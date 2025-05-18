import os
import platform
import subprocess
import sys
from flask import Flask, render_template, request, send_from_directory

# ─── PATH SETUP ─────────────────────────────────────────────────────
BASE_DIR      = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(BASE_DIR, '..'))
TEMPLATE_DIR  = os.path.join(PROJECT_ROOT, 'templates')
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'static', 'uploads')

# ─── FLASK APP ──────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=None                 # เราจะเสิร์ฟ static เอง
)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ─── serve style.css (ไฟล์อยู่ใน templates/) ──────────────────────
@app.route('/style.css')
def style_css():
    return send_from_directory(TEMPLATE_DIR, 'style.css')

# ─── serve อัปโหลดไฟล์รูป (ให้ <img> ใช้งาน) ─────────────────────
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ─── helper: รัน dem.py ด้วย interpreter ปัจจุบัน (3.10.4) ───────
def run_dem(img_path: str):
    py     = sys.executable
    script = os.path.join(PROJECT_ROOT, 'modules', 'dem', 'dem.py')
    out    = subprocess.check_output([py, script, img_path], text=True).strip()
    kind, sigma_str = out.split()
    sigma = None if sigma_str == 'None' else float(sigma_str)
    return kind, sigma

# ─── ROUTES ─────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/restore', methods=['POST'])
def restore():
    file = request.files.get('image')
    if not file:
        return "กรุณาเลือกไฟล์", 400

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    in_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(in_path)

    # === เรียก DEM ===
    kind, sigma = run_dem(in_path)
    if kind == 'blur':
        msg = "ตรวจพบภาพเบลอ → ควรใช้โมเดล DeblurGANv2"
    elif kind == 'noise':
        msg = f"ตรวจพบภาพมีนอยส์ (σ ≈ {sigma:.1f}) → ควรใช้โมเดล FFDNet"
    else:
        msg = "ภาพความละเอียดต่ำ → ควรใช้โมเดล EDSR (Super-Resolution)"

    return render_template(
        'result.html',
        filename=os.path.basename(in_path),
        message=msg
    )

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

# ─── MAIN ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
