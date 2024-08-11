import random
from django.db import models
from PIL import Image
from io import BytesIO
import uuid
from django.core.files.base import ContentFile
import fitz
from django.contrib.auth.models import AbstractUser

# Create your models here.

class User(AbstractUser):
    email = models.EmailField(unique=True)
    contact = models.CharField(max_length = 10, null=True, blank=True)
    photo = models.ImageField(upload_to='user_photos', null=True, blank=True)
    
    class Meta:
        db_table = 'auth_user'

class VerificationEmail(models.Model):
    id = models.UUIDField(editable=False, default = uuid.uuid4, primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_on = models.DateTimeField(auto_now_add=True)
    status = models.BooleanField(default=False)

class FileInput(models.Model):
    file = models.FileField(upload_to='pdf_uploads')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created= models.DateTimeField(auto_now_add=True)
    
class FilePage(models.Model):
    file = models.ForeignKey(FileInput, on_delete=models.CASCADE, null=True, blank=True)
    image = models.ImageField(upload_to='Output_images')
    created= models.DateTimeField(auto_now_add=True)
    
    @classmethod
    def save_image_from_pixmap(self, pixmap, page_no,file:FileInput):
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        image_io = BytesIO()
        image.save(image_io, format='JPEG')

        if file:
            file_page = self.objects.create(file=file)
            filename = file.file.name.split('.')[0]
            file_page.image.save(f'{filename}_{page_no}.jpg', ContentFile(image_io.getvalue()))
        return file_page

class ImageInput(models.Model):
    image = models.ImageField(upload_to= 'image_uploads')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created= models.DateTimeField(auto_now_add=True)
    table = models.JSONField(blank = True, null=True)

class WordConversion(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='image_uploads')
    document = models.FileField(upload_to='converted_document')
    
class ScannedFile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to= 'scanned_files')
    created= models.DateTimeField(auto_now_add=True)

class ScannedImage(models.Model):
    file = models.ForeignKey(ScannedFile, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='scanned_images')
    created= models.DateTimeField(auto_now_add=True)
    
    def save_image(self, image_file):        
        pil_image = Image.fromarray(image_file)
        image_io = BytesIO()
        pil_image.save(image_io, format='PNG')
        image_io.seek(0)
        self.image.save('scanned_image.png', image_io)
        
class GuestScannedFile(models.Model):
    # user = models.ForeignKey(User, on_delete=models.CASCADE)
    identifier = models.CharField(max_length=255)
    file = models.FileField(upload_to= 'scanned_files')
    created= models.DateTimeField(auto_now_add=True)

class GuestScannedImage(models.Model):
    file = models.ForeignKey(GuestScannedFile, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='scanned_images')
    created= models.DateTimeField(auto_now_add=True)
    
    def save_image(self, image_file):        
        pil_image = Image.fromarray(image_file)
        image_io = BytesIO()
        pil_image.save(image_io, format='PNG')
        image_io.seek(0)
        self.image.save('guest_scanned_image.png', image_io)
    
class PasswordResetRequest(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    code = models.IntegerField(default=None, blank=True, null=True, unique=True)
    
    def save(self, *args, **kwargs):
        if not self.code:
            self.code = self.generate_unique_code()
        super().save(*args, **kwargs)

    def generate_unique_code(self):
        while True:
            code = random.randint(1000, 9999)
            if not PasswordResetRequest.objects.filter(code=code).exists():
                return code
    
class BugReport(models.Model):
    description = models.TextField(blank=True, null=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
class BugImage(models.Model):
    report = models.ForeignKey(BugReport, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='bug_images')

class ContactForm(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField()
    subject = models.CharField(max_length=255, null=True, blank=True)
    description = models.TextField()
    
    