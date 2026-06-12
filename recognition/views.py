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
from .forms import ImageUploadForm
from .utils import predict_image, CLASS_NAMES, CLASS_NAMES_CN


# ============ 视图函数 ============

def upload(request):
    """
    显示图片上传表单页面，作为系统首页
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
    # 检查是否为POST请求且包含图片文件
    if request.method == 'POST' and request.FILES.get('image'):
        form = ImageUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            uploaded_file = request.FILES['image']# 获取上传的文件对象

            fs = FileSystemStorage()#默认保存到media目录
            filename = fs.save(uploaded_file.name, uploaded_file)# 保存文件并获取文件名
            file_path = fs.path(filename)# 获取文件的完整路径
            
            # 获取当前事件循环，用于异步任务调度
            loop = asyncio.get_event_loop()
            
            # 异步执行模型预测任务
            # 使用 run_in_executor 将同步的 predict_image 函数提交到线程池执行
            # None 表示使用默认的 ThreadPoolExecutor
            # file_path 是要预测的图片文件路径
            result = await loop.run_in_executor(
                None,           # executor参数，None表示使用默认线程池
                predict_image,  # 要执行的函数（来自utils.py）
                file_path       # 传递给predict_image的参数
            )
            
            # 构建模板渲染所需的上下文数据
            # 将预测结果转换为模板可用的格式
            context = {
                # 模型1（MobileNetV2）的预测结果
                'label1': result['label1'],        # 中文标签（如"黄豆"）
                'label1_code': result['label1_code'],  # 英文标签代码（如"huangdou"）
                'conf1': f'{result["confidence1"]:.2%}',  # 置信度百分比格式
                'result1': result['result1'],      # 识别结果（"是"或"不是"）
                'conf1_num': f'{result["confidence1"]*100:.1f}',  # 置信度数字（0-100）

                # 模型2（ResNet50）的预测结果
                'label2': result['label2'],
                'label2_code': result['label2_code'],
                'conf2': f'{result["confidence2"]:.2%}',
                'result2': result['result2'],
                'conf2_num': f'{result["confidence2"]*100:.1f}',

                # 上传图片的访问URL（供模板显示）
                'img_url': fs.url(filename),
                # 模型是否成功加载（用于调试/状态显示）
                'model_loaded': result['model_loaded']
            }  
            
            return render(request, 'result.html', context)
        else:
            errors = form.errors.get('image', '请上传有效的图片文件')
            return render(request, 'upload.html', {'error': errors})
    
    return render(request, 'upload.html')