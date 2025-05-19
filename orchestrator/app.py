# orchestrator/app.py
import os
import subprocess
import sys
import uuid
from flask import Flask, render_template, request, send_from_directory

# ─── PATH SETUP ─────────────────────────────────────────────────────
BASE_DIR     = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)  # ให้ import modules.arf.run_arf ได้

TEMPLATE_DIR  = os.path.join(PROJECT_ROOT, 'templates')
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'static', 'uploads')

# ─── IMPORT RUN_ARF ─────────────────────────────────────────────────
from modules.arf.run_arf import apply_arf

# ─── FLASK APP ──────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=None
)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ─── serve style.css & รูปอัปโหลด ──────────────────────────────────
@app.route('/style.css')
def style_css():
    return send_from_directory(TEMPLATE_DIR, 'style.css')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ─── helper: DEM วิเคราะห์ชนิดภาพ ──────────────────────────────────
def run_dem(img_path: str):
    py     = sys.executable
    script = os.path.join(PROJECT_ROOT, 'modules', 'dem', 'dem.py')
    kind, sigma_str = subprocess.check_output(
        [py, script, img_path], text=True
    ).strip().split()
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

    kind, sigma = run_dem(in_path)
    if kind == 'blur':
        msg = "ตรวจพบภาพเบลอ → ควรใช้โมเดล DeblurGAN-v2"
    elif kind == 'noise':
        msg = f"ตรวจพบภาพมีนอยส์ (σ ≈ {sigma:.1f}) → ควรใช้โมเดล FFDNet"
    else:
        msg = "ภาพความละเอียดต่ำ → ควรใช้โมเดล EDSR (Super-Resolution)"

    return render_template(
        'result.html',
        filename=os.path.basename(in_path),
        message=msg,
        kind=kind,
        sigma=sigma if sigma else '',
        processed=False      # ยังไม่กดดำเนินแก้ไข
    )

@app.route('/process', methods=['POST'])
def process_image():
    filename = request.form['filename']
    kind     = request.form['kind']
    sigma    = request.form.get('sigma')

    inp_path = os.path.join(UPLOAD_FOLDER, filename)

    # รองรับ blur → DeblurGAN-v2
    if 'blur' in kind:
        out_path = apply_arf(inp_path, kind, sigma)
        out_name = os.path.basename(out_path)
        return render_template(
            'result.html',
            filename=out_name,
            message='✓ แก้ภาพเบลอด้วย DeblurGAN-v2 แล้ว',
            kind=kind,
            sigma=sigma,
            processed=True       # กดแล้ว → ให้ซ่อนปุ่มดำเนิน และโชว์ดาวน์โหลด
        )

    # รองรับ noise → FFDNet
    if 'noise' in kind:
        out_path = apply_arf(inp_path, kind, sigma)
        out_name = os.path.basename(out_path)
        return render_template(
            'result.html',
            filename=out_name,
            message=f'✓ แก้ภาพมีนอยส์ (σ ≈ {float(sigma):.1f}) ด้วย FFDNet แล้ว',
            kind=kind,
            sigma=sigma,
            processed=True
        )

    # ชนิดอื่นยังไม่รองรับ
    return render_template(
        'result.html',
        filename=filename,
        message='(ยังไม่รองรับชนิดนี้)',
        kind=kind,
        sigma=sigma,
        processed=True
    )

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

# ─── MAIN ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
