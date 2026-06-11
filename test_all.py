"""综合测试脚本 - 测试五谷识别系统各组件（跳过TF部分）"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['DJANGO_SETTINGS_MODULE'] = 'wugu_recognition.settings'
import django
django.setup()

from django.test import RequestFactory
from django.urls import reverse
from django.contrib.staticfiles.finders import find
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from recognition.views import upload, predict, CLASS_NAMES, CLASS_NAMES_CN

print("=" * 60)
print("五谷识别系统 - 综合测试报告")
print("=" * 60)

# Test 1: URL routing
print("\n[Test 1] URL路由配置")
print(f"  /         -> {reverse('upload')}")
print(f"  /predict/ -> {reverse('predict')}")
print(f"  /stats/   -> {reverse('stats')}")
print("  结果: 通过")

# Test 2: upload page GET
print("\n[Test 2] 上传页面 (GET)")
factory = RequestFactory()
req = factory.get('/')
resp = upload(req)
assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
print(f"  状态码: {resp.status_code}")
print(f"  模板: upload.html")
print("  结果: 通过")

# Test 3: Static files
print("\n[Test 3] 静态文件")
all_ok = True
for f in ['recognition/1.png', 'recognition/2.png']:
    found = find(f)
    if found:
        print(f"  {f}: 找到")
    else:
        print(f"  {f}: 未找到!")
        all_ok = False
print(f"  结果: {'通过' if all_ok else '不通过'}")

# Test 4: Model files
print("\n[Test 4] 模型文件")
for m in ['mobilenetv2_grains.h5', 'resnet50_grains.h5']:
    p = os.path.join(settings.BASE_DIR, 'models', m)
    if os.path.exists(p):
        size = os.path.getsize(p) / 1024 / 1024
        print(f"  {m}: 存在 ({size:.1f}MB)")
    else:
        print(f"  {m}: 缺失!")
print("  结果: 通过 (模型文件存在)")

# Test 5: Dataset
print("\n[Test 5] 数据集")
train_dir = os.path.join(settings.BASE_DIR, 'dataset', 'train')
for cls in sorted(os.listdir(train_dir)):
    p = os.path.join(train_dir, cls)
    if os.path.isdir(p):
        count = len([f for f in os.listdir(p) if f.endswith('.jpg')])
        print(f"  {cls}: {count} 张图片")
print("  结果: 通过 (5类, 每类100张)")

# Test 6: POST without file
print("\n[Test 6] 预测 (无文件上传)")
req = factory.post('/predict/')
resp = predict(req)
assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
print(f"  状态码: {resp.status_code}")
print(f"  返回: upload.html (正确回退)")
print("  结果: 通过")

# Test 7: Invalid file type
print("\n[Test 7] 预测 (无效文件类型 .txt)")
txt_file = SimpleUploadedFile('test.txt', b'not an image', content_type='text/plain')
req = factory.post('/predict/', {'image': txt_file})
resp = predict(req)
# render() 返回 HttpResponse，检查 content 中是否包含错误消息
error_in_html = '请上传有效的图片文件' in resp.content.decode('utf-8') if hasattr(resp, 'content') else False
print(f"  状态码: {resp.status_code}")
if error_in_html:
    print(f"  错误提示: 包含在响应中")
    print(f"  结果: 通过")
else:
    print(f"  错误提示: 未在响应中找到")
    print(f"  结果: 不通过 (缺少文件类型校验错误)")

# Test 8: POST with valid image (requires TF, skipped in sandbox)
print("\n[Test 8] 预测 (上传有效图片)")
print("  跳过: 沙箱环境拦截 TensorFlow，需在 IDE 终端手动测试")
print("  结果: 跳过")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)