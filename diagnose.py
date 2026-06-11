"""诊断脚本：测试模型在训练数据上的预测表现"""
import os, sys, numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else r'd:\HuaweiMoveData\Users\hh\Desktop\作业\wugu_recognition'
sys.path.insert(0, os.path.join(BASE_DIR, '.venv_libs'))
os.environ['KERAS_HOME'] = os.path.join(BASE_DIR, '.keras_cache')
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

CLASS_NAMES = ['huangdou', 'shuidao', 'xiaomai', 'xiaomi', 'yumi']
CLASS_CN = {'huangdou':'黄豆','shuidao':'稻谷','xiaomai':'小麦','xiaomi':'小米','yumi':'玉米'}

print("加载模型...")
m1 = load_model(os.path.join(BASE_DIR, 'models', 'mobilenetv2_grains.h5'))
m2 = load_model(os.path.join(BASE_DIR, 'models', 'resnet50_grains.h5'))
print(f"MobileNetV2 输出层: {m1.output_shape}")
print(f"ResNet50 输出层: {m2.output_shape}")
print()

# 测试每个类别的第一张图片
for cls in CLASS_NAMES:
    d = os.path.join(BASE_DIR, 'dataset', 'train', cls)
    if not os.path.isdir(d):
        print(f"目录不存在: {d}")
        continue
    files = sorted([f for f in os.listdir(d) if f.endswith('.jpg')])
    if not files:
        print(f"无图片: {cls}")
        continue
    f = os.path.join(d, files[0])
    img = image.load_img(f, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0

    p1 = m1.predict(arr, verbose=0)[0]
    p2 = m2.predict(arr, verbose=0)[0]

    top1 = np.argsort(p1)[-3:][::-1]
    top2 = np.argsort(p2)[-3:][::-1]

    print(f'真实: {CLASS_CN[cls]} ({cls})')
    print(f'  MobileNetV2: pred={CLASS_CN[CLASS_NAMES[np.argmax(p1)]]} conf={np.max(p1):.4f}')
    print(f'    top3: {[(CLASS_CN[CLASS_NAMES[i]], f"{p1[i]:.4f}") for i in top1]}')
    print(f'  ResNet50:    pred={CLASS_CN[CLASS_NAMES[np.argmax(p2)]]} conf={np.max(p2):.4f}')
    print(f'    top3: {[(CLASS_CN[CLASS_NAMES[i]], f"{p2[i]:.4f}") for i in top2]}')
    print()

# 整体准确率统计
print("=" * 50)
print("整体准确率统计 (每类取前5张测试)")
print("=" * 50)
correct1 = 0
correct2 = 0
total = 0
for cls in CLASS_NAMES:
    d = os.path.join(BASE_DIR, 'dataset', 'train', cls)
    if not os.path.isdir(d):
        continue
    files = sorted([f for f in os.listdir(d) if f.endswith('.jpg')])[:5]
    for fn in files:
        f = os.path.join(d, fn)
        img = image.load_img(f, target_size=(224, 224))
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0) / 255.0
        p1 = m1.predict(arr, verbose=0)[0]
        p2 = m2.predict(arr, verbose=0)[0]
        total += 1
        if CLASS_NAMES[np.argmax(p1)] == cls:
            correct1 += 1
        if CLASS_NAMES[np.argmax(p2)] == cls:
            correct2 += 1

print(f"MobileNetV2: {correct1}/{total} = {correct1/total*100:.1f}%")
print(f"ResNet50:    {correct2}/{total} = {correct2/total*100:.1f}%")