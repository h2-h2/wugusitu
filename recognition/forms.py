# 上传图片
from django import forms
from django.core.validators import FileExtensionValidator

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'gif']


class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        label='上传图片',
        validators=[
            FileExtensionValidator(allowed_extensions=ALLOWED_EXTENSIONS, 
                                  message='请上传有效的图片文件（JPG/JPEG/PNG/BMP/GIF）')
        ]
    )