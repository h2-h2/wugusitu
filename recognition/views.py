"""
五谷识别系统 - Django视图模块
功能：处理图片上传、双模型预测、结果展示

视图函数说明：
- upload(): 首页，展示图片上传表单
- predict(): 异步处理图片上传和预测，返回识别结果
"""

# ============ 导入依赖库 ============
import asyncio
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from recognition.forms import ImageUploadForm
from recognition.utils import predict_image, CLASS_NAMES, CLASS_NAMES_CN


# ============ 视图函数 ============

def upload(request):
    """
    上传页面视图
    
    显示图片上传表单页面，作为系统首页
    
    参数：
        request: Django HTTP请求对象
        
    返回：
        渲染后的上传页面模板 (upload.html)
    """
    return render(request, 'upload.html')


async def predict(request):
    """
    异步预测视图函数
    
    处理用户上传的图片，异步执行双模型预测，返回识别结果
    
    主要流程：
    1. 验证请求方法为POST且包含图片文件
    2. 使用ImageUploadForm验证文件类型
    3. 保存上传的图片到media目录
    4. 使用asyncio.run_in_executor异步调用predict_image()
    5. 准备上下文数据渲染结果页面
    
    参数：
        request: Django HTTP请求对象（包含POST数据和上传文件）
        
    返回：
        渲染后的结果页面模板(result.html)或上传页面模板(upload.html)
    """
    if request.method == 'POST' and request.FILES.get('image'):
        form = ImageUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            uploaded_file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = fs.path(filename)
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                predict_image, 
                file_path
            )
            
            context = {
                'label1': result['label1'],
                'label1_code': result['label1_code'],
                'conf1': f'{result["confidence1"]:.2%}',
                'result1': result['result1'],
                'conf1_num': f'{result["confidence1"]*100:.1f}',
                'label2': result['label2'],
                'label2_code': result['label2_code'],
                'conf2': f'{result["confidence2"]:.2%}',
                'result2': result['result2'],
                'conf2_num': f'{result["confidence2"]*100:.1f}',
                'img_url': fs.url(filename),
                'model_loaded': result['model_loaded']
            }
            
            return render(request, 'result.html', context)
        else:
            errors = form.errors.get('image', '请上传有效的图片文件')
            return render(request, 'upload.html', {'error': errors})
    
    return render(request, 'upload.html')