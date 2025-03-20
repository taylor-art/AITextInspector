from flask import Flask, request, render_template, jsonify
import os
import logging
from .detector import AITextDetectorApp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_app(model_path, model_name="desklib/ai-text-detector-v1.01", static_folder="static", template_folder="templates"):
    """
    创建Flask应用
    
    Args:
        model_path: 模型路径
        model_name: 预训练模型名称
        static_folder: 静态文件夹
        template_folder: 模板文件夹
        
    Returns:
        Flask应用
    """
    app = Flask(__name__, static_folder=static_folder, template_folder=template_folder)
    
    # 初始化检测器
    detector = AITextDetectorApp(model_path=model_path, model_name=model_name)
    
    @app.route('/')
    def index():
        """主页"""
        return render_template('index.html')
    
    @app.route('/detect', methods=['POST'])
    def detect():
        """处理文本检测请求"""
        try:
            # 获取文本
            data = request.json
            text = data.get('text', '')
            
            if not text:
                return jsonify({
                    'success': False,
                    'error': '文本不能为空'
                }), 400
            
            # 检测文本
            result = detector.detect(text)
            
            return jsonify({
                'success': True,
                'result': {
                    'is_ai_generated': result['is_ai_generated'],
                    'confidence': result['confidence'],
                    'text_type': 'AI生成' if result['is_ai_generated'] else '人类撰写'
                }
            })
        except Exception as e:
            logger.error(f"检测文本时出错: {e}")
            return jsonify({
                'success': False,
                'error': f'检测文本时出错: {str(e)}'
            }), 500
    
    @app.route('/detect_file', methods=['POST'])
    def detect_file():
        """处理文件检测请求"""
        try:
            # 获取文件
            if 'file' not in request.files:
                return jsonify({
                    'success': False,
                    'error': '没有上传文件'
                }), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': '未选择文件'
                }), 400
            
            # 检查文件类型
            allowed_extensions = {'txt', 'docx', 'pdf'}
            if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
                return jsonify({
                    'success': False,
                    'error': '不支持的文件类型，仅支持txt、docx和pdf'
                }), 400
            
            # 保存文件
            file_path = os.path.join('uploads', file.filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(file_path)
            
            # 读取文件内容
            text = ""
            if file.filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif file.filename.endswith('.docx'):
                # 需要安装python-docx库
                from docx import Document
                document = Document(file_path)
                text = '\n'.join([paragraph.text for paragraph in document.paragraphs])
            elif file.filename.endswith('.pdf'):
                # 需要安装PyPDF2库
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
            
            # 检测文本
            result = detector.detect(text)
            
            # 删除临时文件
            os.remove(file_path)
            
            return jsonify({
                'success': True,
                'result': {
                    'is_ai_generated': result['is_ai_generated'],
                    'confidence': result['confidence'],
                    'text_type': 'AI生成' if result['is_ai_generated'] else '人类撰写',
                    'filename': file.filename
                }
            })
        except Exception as e:
            logger.error(f"检测文件时出错: {e}")
            return jsonify({
                'success': False,
                'error': f'检测文件时出错: {str(e)}'
            }), 500
    
    @app.route('/batch_detect', methods=['POST'])
    def batch_detect():
        """处理批量文本检测请求"""
        try:
            # 获取文本列表
            data = request.json
            texts = data.get('texts', [])
            
            if not texts:
                return jsonify({
                    'success': False,
                    'error': '文本列表不能为空'
                }), 400
            
            # 批量检测文本
            results = detector.batch_detect(texts)
            
            # 格式化结果
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append({
                    'id': i,
                    'is_ai_generated': result['is_ai_generated'],
                    'confidence': result['confidence'],
                    'text_type': 'AI生成' if result['is_ai_generated'] else '人类撰写'
                })
            
            return jsonify({
                'success': True,
                'results': formatted_results
            })
        except Exception as e:
            logger.error(f"批量检测文本时出错: {e}")
            return jsonify({
                'success': False,
                'error': f'批量检测文本时出错: {str(e)}'
            }), 500
    
    @app.errorhandler(404)
    def page_not_found(e):
        """处理404错误"""
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def server_error(e):
        """处理500错误"""
        return render_template('500.html'), 500
    
    return app 