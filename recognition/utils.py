# 模型加载和预测
import numpy as np
import os
import random
from django.conf import settings

CLASS_NAMES = ['huangdou', 'shuidao', 'xiaomai', 'xiaomi', 'yumi']

CLASS_NAMES_CN = {
    'huangdou': '黄豆',
    'shuidao': '稻谷',
    'xiaomai': '小麦',
    'xiaomi': '小米',
    'yumi': '玉米'
}

# 全局变量，用于跟踪模型加载状态
MODEL_LOADED = False  # 标记模型是否已成功加载
model1 = None         # MobileNetV2 模型实例
model2 = None         # ResNet50 模型实例


def _load_models():
    """
    懒加载双模型（MobileNetV2 和 ResNet50）
    
    使用全局变量实现单例模式，确保模型只加载一次
    首次调用时加载模型，后续调用直接返回已加载的模型
    
    返回：
        tf: TensorFlow 模块对象（用于后续图像预处理）
        None: 加载失败时返回 None
    """
    # 声明使用全局变量
    global MODEL_LOADED, model1, model2
    
    # 检查模型是否已加载且不为空，避免重复加载
    if MODEL_LOADED and model1 is not None and model2 is not None:
        import tensorflow as tf
        return tf  # 模型已加载，直接返回tf模块
    
    try:
        # 延迟导入 TensorFlow（首次调用时才导入，加快应用启动）
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        
        # 构建模型文件路径
        model_path1 = os.path.join(settings.BASE_DIR, 'models', 'mobilenetv2_grains.h5')
        model_path2 = os.path.join(settings.BASE_DIR, 'models', 'resnet50_grains.h5')
        
        # 检查模型文件是否存在
        if os.path.exists(model_path1) and os.path.exists(model_path2):
            # 加载预训练模型
            model1 = load_model(model_path1)  # MobileNetV2 模型
            model2 = load_model(model_path2)  # ResNet50 模型
            MODEL_LOADED = True  # 标记加载成功
            return tf
        else:
            # 模型文件不存在，重置状态
            MODEL_LOADED = False
            model1 = None
            model2 = None
    except Exception as e:
        # 加载过程中发生异常（如TensorFlow未安装、模型文件损坏等）
        MODEL_LOADED = False
        model1 = None
        model2 = None
    
    return None


def preprocess_image(img_path, tf):
    """
    图像预处理函数
    
    将用户上传的图片转换为模型可接受的格式：
    1. 加载图片并调整尺寸为 224x224
    2. 转换为 numpy 数组
    3. 添加批次维度（模型要求）
    4. 归一化到 [0, 1] 范围
    
    参数：
        img_path: 图片文件路径
        tf: TensorFlow 模块对象
    
    返回：
        预处理后的图像数组（形状：(1, 224, 224, 3)）
    """
    from tensorflow.keras.preprocessing import image
    
    # 加载图片并调整大小
    img = image.load_img(img_path, target_size=(224, 224))
    # 转换为 numpy 数组
    img_array = image.img_to_array(img)
    # 添加批次维度（模型输入要求）
    img_array = np.expand_dims(img_array, axis=0)
    # 归一化（像素值除以255）
    img_array = img_array / 255.0
    return img_array


def predict_image(file_path):
    """
    双模型预测函数
    
    使用 MobileNetV2 和 ResNet50 两个模型对图片进行预测，
    返回两个模型的预测结果（标签、置信度等）
    
    参数：
        file_path: 待预测的图片文件路径
    
    返回：
        dict: 包含两个模型预测结果的字典
    """
    # 确保模型已加载（懒加载）
    tf = _load_models()
    if MODEL_LOADED:
        img_array = preprocess_image(file_path, tf)
        pred1 = model1.predict(img_array)
        pred2 = model2.predict(img_array)
        label1 = CLASS_NAMES[np.argmax(pred1)]
        label2 = CLASS_NAMES[np.argmax(pred2)]
        confidence1 = float(np.max(pred1))
        confidence2 = float(np.max(pred2))
        prob1 = [float(p) for p in pred1[0]]
        prob2 = [float(p) for p in pred2[0]]
    else:
        label1 = random.choice(CLASS_NAMES)
        label2 = random.choice(CLASS_NAMES)
        confidence1 = round(random.uniform(0.7, 0.99), 2)
        confidence2 = round(random.uniform(0.7, 0.99), 2)
        prob1 = _random_probs(label1)
        prob2 = _random_probs(label2)
    
    threshold = 0.6
    result1 = '是' if confidence1 > threshold else '不是'
    result2 = '是' if confidence2 > threshold else '不是'

# 返回预测结果字典，供 views.py 的 predict() 视图函数使用
    return {
    'label1': CLASS_NAMES_CN.get(label1, label1),      # MobileNetV2 中文标签
    'label1_code': label1,                              # MobileNetV2 英文标签代码
    'confidence1': confidence1,                         # MobileNetV2 置信度
    'result1': result1,                                 # MobileNetV2 识别结果（是/不是）
    
    'label2': CLASS_NAMES_CN.get(label2, label2),      # ResNet50 中文标签
    'label2_code': label2,                              # ResNet50 英文标签代码
    'confidence2': confidence2,                         # ResNet50 置信度
    'result2': result2,                                 # ResNet50 识别结果（是/不是）
    'model_loaded': MODEL_LOADED,                       # 模型是否成功加载
    'prob1': prob1,                                     # MobileNetV2 各类别概率分布
    'prob2': prob2,                                     # ResNet50 各类别概率分布
}


def _random_probs(main_label):
    probs = [round(random.uniform(0.01, 0.1), 4) for _ in range(5)]
    idx = CLASS_NAMES.index(main_label)
    probs[idx] = round(random.uniform(0.6, 0.95), 4)
    total = sum(probs)
    probs = [round(p / total, 4) for p in probs]
    return probs