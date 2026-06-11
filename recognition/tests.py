from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from recognition.forms import ImageUploadForm, ALLOWED_EXTENSIONS
from recognition.utils import (
    _load_models, 
    predict_image, 
    CLASS_NAMES, 
    CLASS_NAMES_CN,
    MODEL_LOADED
)
import os
import tempfile
from PIL import Image


class ImageUploadFormTest(TestCase):
    """测试图片上传表单验证功能"""
    
    def test_valid_image_formats(self):
        """测试所有允许的图片格式都能通过验证"""
        for ext in ALLOWED_EXTENSIONS:
            fd, temp_path = tempfile.mkstemp(suffix=f'.{ext}')
            os.close(fd)
            try:
                img = Image.new('RGB', (100, 100), color='red')
                img.save(temp_path)
                with open(temp_path, 'rb') as f:
                    uploaded_file = SimpleUploadedFile(
                        f'test.{ext}', 
                        f.read(), 
                        content_type=f'image/{ext}'
                    )
                form = ImageUploadForm(files={'image': uploaded_file})
                self.assertTrue(form.is_valid(), f"Form should accept {ext} format")
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    def test_invalid_image_format(self):
        """测试不允许的图片格式被正确拒绝"""
        invalid_extensions = ['txt', 'pdf', 'doc', 'zip', 'exe']
        for ext in invalid_extensions:
            fd, temp_path = tempfile.mkstemp(suffix=f'.{ext}')
            os.close(fd)
            try:
                with open(temp_path, 'wb') as f:
                    f.write(b'test content')
                with open(temp_path, 'rb') as f:
                    uploaded_file = SimpleUploadedFile(
                        f'test.{ext}', 
                        f.read(), 
                        content_type=f'application/{ext}'
                    )
                form = ImageUploadForm(files={'image': uploaded_file})
                self.assertFalse(form.is_valid(), f"Form should reject {ext} format")
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    def test_empty_file(self):
        """测试空文件被正确拒绝"""
        uploaded_file = SimpleUploadedFile('empty.jpg', b'', content_type='image/jpeg')
        form = ImageUploadForm(files={'image': uploaded_file})
        self.assertFalse(form.is_valid())
    
    def test_missing_image_field(self):
        """测试缺少图片字段时表单无效"""
        form = ImageUploadForm(files={})
        self.assertFalse(form.is_valid())
        self.assertIn('image', form.errors)


class ModelLoadingTest(TestCase):
    """测试模型加载功能"""
    
    def test_model_load_function_exists(self):
        """测试模型加载函数存在"""
        self.assertTrue(callable(_load_models))
    
    def test_class_names_defined(self):
        """测试类别名称已定义"""
        self.assertIsInstance(CLASS_NAMES, list)
        self.assertEqual(len(CLASS_NAMES), 5)
    
    def test_class_names_cn_defined(self):
        """测试中文类别名称已定义"""
        self.assertIsInstance(CLASS_NAMES_CN, dict)
        for name in CLASS_NAMES:
            self.assertIn(name, CLASS_NAMES_CN)


class PredictImageTest(TestCase):
    """测试图片预测功能"""
    
    def test_predict_image_function_exists(self):
        """测试预测函数存在"""
        self.assertTrue(callable(predict_image))
    
    def test_predict_returns_dict(self):
        """测试预测返回字典格式"""
        fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
        try:
            img = Image.new('RGB', (224, 224), color='red')
            img.save(temp_path)
            result = predict_image(temp_path)
            self.assertIsInstance(result, dict)
            required_fields = [
                'label1', 'label1_code', 'confidence1', 'result1',
                'label2', 'label2_code', 'confidence2', 'result2',
                'model_loaded'
            ]
            for field in required_fields:
                self.assertIn(field, result, f"Result should contain '{field}'")
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def test_predict_confidence_range(self):
        """测试置信度在合理范围内"""
        fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
        try:
            img = Image.new('RGB', (224, 224), color='blue')
            img.save(temp_path)
            result = predict_image(temp_path)
            self.assertGreaterEqual(result['confidence1'], 0)
            self.assertLessEqual(result['confidence1'], 1)
            self.assertGreaterEqual(result['confidence2'], 0)
            self.assertLessEqual(result['confidence2'], 1)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def test_predict_label_code_valid(self):
        """测试标签代码在允许的类别列表中"""
        fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
        try:
            img = Image.new('RGB', (224, 224), color='green')
            img.save(temp_path)
            result = predict_image(temp_path)
            self.assertIn(result['label1_code'], CLASS_NAMES)
            self.assertIn(result['label2_code'], CLASS_NAMES)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def test_predict_result_value(self):
        """测试预测结果值只能是'是'或'不是'"""
        fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
        try:
            img = Image.new('RGB', (224, 224), color='yellow')
            img.save(temp_path)
            result = predict_image(temp_path)
            self.assertIn(result['result1'], ['是', '不是'])
            self.assertIn(result['result2'], ['是', '不是'])
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def test_predict_returns_prob_arrays(self):
        """测试预测返回完整概率数组"""
        fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
        try:
            img = Image.new('RGB', (224, 224), color='red')
            img.save(temp_path)
            result = predict_image(temp_path)
            self.assertIn('prob1', result)
            self.assertIn('prob2', result)
            self.assertEqual(len(result['prob1']), 5)
            self.assertEqual(len(result['prob2']), 5)
            for p in result['prob1']:
                self.assertIsInstance(p, float)
            for p in result['prob2']:
                self.assertIsInstance(p, float)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def test_prob_arrays_sum_near_one(self):
        """测试概率数组之和接近1"""
        fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
        try:
            img = Image.new('RGB', (224, 224), color='blue')
            img.save(temp_path)
            result = predict_image(temp_path)
            total1 = sum(result['prob1'])
            total2 = sum(result['prob2'])
            if result['model_loaded']:
                self.assertAlmostEqual(total1, 1.0, delta=0.1)
                self.assertAlmostEqual(total2, 1.0, delta=0.1)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
