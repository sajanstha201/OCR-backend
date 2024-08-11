from rest_framework import serializers
from ..models import *
from rest_framework import permissions

class FileInputSerializer(serializers.ModelSerializer):
    pages = serializers.SerializerMethodField()
    
    def get_pages(self, obj):
        return (FilePageSerializer(obj.filepage_set.all(), many=True).data)
        
    class Meta:
        model=FileInput
        fields = '__all__'

class FilePageSerializer(serializers.ModelSerializer):

    class Meta:
        model=FilePage
        fields = '__all__'
        
class ImageInputSerializer(serializers.ModelSerializer):
    file = serializers.SerializerMethodField()
    
    def get_file(self, obj):
        request = self.context.get('request')
        try:
            return request.build_absolute_uri(obj.image.url)
        except:
            return obj.image.url
        
    class Meta:
        model=ImageInput
        fields = '__all__'

class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ['id', 'username', 'password', 'email', 'first_name', 'last_name','photo','contact']

class ScannedImageSerializer(serializers.ModelSerializer):
    image = serializers.SerializerMethodField()
    
    class Meta:
        model=ScannedImage
        fields = '__all__'
        
    def get_image(self, obj):
        request = self.context.get('request')
        if request is not None and obj.image:
            try:
                return request.build_absolute_uri(obj.image.url)
            except:
                return obj.image.url


class ScannedFileSerializer(serializers.ModelSerializer):
    file = serializers.FileField(required=False)
    
    pages = serializers.SerializerMethodField()
    def get_pages(self, obj):
        request = self.context.get('request')
        return ScannedImageSerializer(obj.scannedimage_set.all(), many=True, context = {'request':request}).data
    
    class Meta:
        model=ScannedFile
        fields = '__all__'

class WordConversionSerializer(serializers.ModelSerializer):
    file = serializers.SerializerMethodField()
    
    def get_file(self, obj):
        request = self.context.get('request')
        try:
            return (request.build_absolute_uri(obj.document.url))
        except:
            return obj.document.url
    
    class Meta:
        model=WordConversion
        fields = '__all__'


class GuestScannedImageSerializer(serializers.ModelSerializer):
    class Meta:
        model=GuestScannedImage
        fields = '__all__'
        
class GuestScannedFileSerializer(serializers.ModelSerializer):
    file = serializers.FileField(required = False)
    
    pages = serializers.SerializerMethodField()
    def get_pages(self, obj):
        return GuestScannedImageSerializer(obj.guestscannedimage_set.all(), many=True).data
    
    class Meta:
        model=GuestScannedFile
        fields = '__all__'
        
        def to_representation(self, instance):
            data = super().to_representation(instance)
            data.pop('identifier', None)
            return data

class GuestScannedFileOutputSerializer(serializers.ModelSerializer):
    file = serializers.FileField(required = False)
    
    pages = serializers.SerializerMethodField()
    def get_pages(self, obj):
        return GuestScannedImageSerializer(obj.guestscannedimage_set.all(), many=True).data

    class Meta:
        model=GuestScannedFile
        fields = ['file','pages' ,'created']

class PasswordResetRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = PasswordResetRequest
        fields = ['user', 'created']

class BugImageSerializer(serializers.ModelSerializer):
    class Meta:
        model=BugImage
        fields = '__all__'

class BugReportSerializer(serializers.ModelSerializer):
    images = serializers.SerializerMethodField()
    
    def get_images(self, obj):
        return BugImageSerializer(obj.bugimage_set.all(), many=True).data
    class Meta:
        model=BugReport
        fields = '__all__'

class ContactFormSerializer(serializers.ModelSerializer):
    class Meta:
        model=ContactForm
        fields = '__all__'
        
            
