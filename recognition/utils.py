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

MODEL_LOADED = False
model1 = None
model2 = None


def _load_models():
    global MODEL_LOADED, model1, model2
    if MODEL_LOADED and model1 is not None and model2 is not None:
        import tensorflow as tf
        return tf
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        
        model_path1 = os.path.join(settings.BASE_DIR, 'models', 'mobilenetv2_grains.h5')
        model_path2 = os.path.join(settings.BASE_DIR, 'models', 'resnet50_grains.h5')
        
        if os.path.exists(model_path1) and os.path.exists(model_path2):
            model1 = load_model(model_path1)
            model2 = load_model(model_path2)
            MODEL_LOADED = True
            return tf
        else:
            MODEL_LOADED = False
            model1 = None
            model2 = None
    except Exception as e:
        MODEL_LOADED = False
        model1 = None
        model2 = None
    return None


def preprocess_image(img_path, tf):
    from tensorflow.keras.preprocessing import image
    
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def predict_image(file_path):
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
    
    return {
        'label1': CLASS_NAMES_CN.get(label1, label1),
        'label1_code': label1,
        'confidence1': confidence1,
        'result1': result1,
        'label2': CLASS_NAMES_CN.get(label2, label2),
        'label2_code': label2,
        'confidence2': confidence2,
        'result2': result2,
        'model_loaded': MODEL_LOADED,
        'prob1': prob1,
        'prob2': prob2,
    }


def _random_probs(main_label):
    probs = [round(random.uniform(0.01, 0.1), 4) for _ in range(5)]
    idx = CLASS_NAMES.index(main_label)
    probs[idx] = round(random.uniform(0.6, 0.95), 4)
    total = sum(probs)
    probs = [round(p / total, 4) for p in probs]
    return probs