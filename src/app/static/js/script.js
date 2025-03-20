// 在文档加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 获取表单和结果元素
    const textForm = document.getElementById('text-form');
    const fileForm = document.getElementById('file-form');
    const textInput = document.getElementById('text-input');
    const fileInput = document.getElementById('file-input');
    const resultCard = document.getElementById('result-card');
    const resultAlert = document.getElementById('result-alert');
    const resultIcon = document.getElementById('result-icon');
    const resultHeading = document.getElementById('result-heading');
    const resultMessage = document.getElementById('result-message');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');
    const loadingOverlay = document.getElementById('loading-overlay');

    // 文本检测表单提交处理
    textForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const text = textInput.value.trim();
        if (!text) {
            showError('请输入要检测的文本');
            return;
        }
        
        // 显示加载指示器
        showLoading();
        
        // 发送检测请求
        fetch('/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        })
        .then(response => response.json())
        .then(data => {
            // 隐藏加载指示器
            hideLoading();
            
            if (data.success) {
                // 显示检测结果
                displayResult(data.result);
            } else {
                // 显示错误信息
                showError(data.error || '检测失败，请重试');
            }
        })
        .catch(error => {
            // 隐藏加载指示器
            hideLoading();
            // 显示错误信息
            showError('请求失败: ' + error.message);
        });
    });
    
    // 文件检测表单提交处理
    fileForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            showError('请选择要检测的文件');
            return;
        }
        
        // 检查文件类型
        const allowedTypes = ['.txt', '.docx', '.pdf'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        if (!allowedTypes.includes(fileExtension)) {
            showError('不支持的文件类型，仅支持txt、docx和pdf');
            return;
        }
        
        // 检查文件大小
        const maxSize = 10 * 1024 * 1024; // 10MB
        if (file.size > maxSize) {
            showError('文件大小超过限制 (10MB)');
            return;
        }
        
        // 显示加载指示器
        showLoading();
        
        // 创建FormData对象
        const formData = new FormData();
        formData.append('file', file);
        
        // 发送检测请求
        fetch('/detect_file', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // 隐藏加载指示器
            hideLoading();
            
            if (data.success) {
                // 显示检测结果
                displayResult(data.result);
            } else {
                // 显示错误信息
                showError(data.error || '检测失败，请重试');
            }
        })
        .catch(error => {
            // 隐藏加载指示器
            hideLoading();
            // 显示错误信息
            showError('请求失败: ' + error.message);
        });
    });
    
    // 显示检测结果
    function displayResult(result) {
        // 设置结果卡片的颜色
        if (result.is_ai_generated) {
            // AI生成文本
            resultAlert.className = 'alert alert-danger d-flex align-items-center';
            resultIcon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" class="bi bi-robot" viewBox="0 0 16 16"><path d="M6 12.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 0 1h-3a.5.5 0 0 1-.5-.5ZM3 8.062C3 6.76 4.235 5.765 5.53 5.886a26.58 26.58 0 0 0 4.94 0C11.765 5.765 13 6.76 13 8.062v1.157a.933.933 0 0 1-.765.935c-.845.147-2.34.346-4.235.346-1.895 0-3.39-.2-4.235-.346A.933.933 0 0 1 3 9.219V8.062Zm4.542-.827a.25.25 0 0 0-.217.068l-.92.9a24.767 24.767 0 0 1-1.871-.183.25.25 0 0 0-.068.495c.55.076 1.232.149 2.02.193a.25.25 0 0 0 .189-.071l.754-.736.847 1.71a.25.25 0 0 0 .404.062l.932-.97a25.286 25.286 0 0 0 1.922-.188.25.25 0 0 0-.068-.495c-.538.074-1.207.145-1.98.189a.25.25 0 0 0-.166.076l-.754.785-.842-1.7a.25.25 0 0 0-.182-.135Z"/><path d="M8.5 1.866a1 1 0 1 0-1 0V3h-2A4.5 4.5 0 0 0 1 7.5V8a1 1 0 0 0-1 1v2a1 1 0 0 0 1 1v1a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-1a1 1 0 0 0 1-1V9a1 1 0 0 0-1-1v-.5A4.5 4.5 0 0 0 10.5 3h-2V1.866ZM14 7.5V13a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V7.5A3.5 3.5 0 0 1 5.5 4h5A3.5 3.5 0 0 1 14 7.5Z"/></svg>';
            resultHeading.innerText = '检测到AI生成文本';
            resultMessage.innerText = '该文本很可能是由AI生成的。';
        } else {
            // 人类撰写文本
            resultAlert.className = 'alert alert-success d-flex align-items-center';
            resultIcon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" class="bi bi-person" viewBox="0 0 16 16"><path d="M8 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm2-3a2 2 0 1 1-4 0 2 2 0 0 1 4 0Zm4 8c0 1-1 1-1 1H3s-1 0-1-1 1-4 6-4 6 3 6 4Zm-1-.004c-.001-.246-.154-.986-.832-1.664C11.516 10.68 10.289 10 8 10c-2.29 0-3.516.68-4.168 1.332-.678.678-.83 1.418-.832 1.664h10Z"/></svg>';
            resultHeading.innerText = '检测到人类撰写文本';
            resultMessage.innerText = '该文本很可能是由人类撰写的。';
        }
        
        // 更新置信度进度条
        const confidencePercent = Math.round(result.confidence * 100);
        confidenceBar.style.width = confidencePercent + '%';
        confidenceBar.setAttribute('aria-valuenow', confidencePercent);
        confidenceBar.innerText = confidencePercent + '%';
        
        // 更新置信度文本
        confidenceText.innerText = '置信度: ' + confidencePercent + '%';
        
        // 如果是文件检测，添加文件名信息
        if (result.filename) {
            resultMessage.innerText += ' (文件: ' + result.filename + ')';
        }
        
        // 显示结果卡片
        resultCard.classList.remove('d-none');
    }
    
    // 显示错误消息
    function showError(message) {
        alert(message);
    }
    
    // 显示加载指示器
    function showLoading() {
        loadingOverlay.classList.remove('d-none');
    }
    
    // 隐藏加载指示器
    function hideLoading() {
        loadingOverlay.classList.add('d-none');
    }
}); 