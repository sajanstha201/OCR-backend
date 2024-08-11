import json
from django.http import HttpResponse, JsonResponse, QueryDict
from .serializers import *
from ..models import *
from rest_framework import viewsets
from django.utils.translation import gettext_lazy as _
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from django.utils.timezone import now as django_now
from rest_framework.response import Response
from rest_framework.decorators import action
from django.contrib.auth import authenticate
from rest_framework import status
from django.conf import settings
from ..views import *
from rest_framework.authentication import TokenAuthentication
from rest_framework.authtoken.models import Token
from rest_framework.authtoken.views import ObtainAuthToken
import datetime
from django.utils import timezone
from django.http import FileResponse
import base64
from datetime import datetime, timedelta
from rest_framework.authentication import BaseAuthentication
import uuid
from rest_framework.permissions import AllowAny
from rest_framework.pagination import PageNumberPagination
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.utils import timezone
from datetime import timedelta
from django.shortcuts import redirect
import socket
from django.db import transaction
from rest_framework.permissions import DjangoModelPermissions, BasePermission
from django.contrib.auth.models import Permission, Group
from django.middleware.csrf import get_token
from django.views.decorators.csrf import csrf_exempt
from rest_framework.authtoken.serializers import AuthTokenSerializer
import time
from ..views import process_image_for_table_extraction

class PostOnlyFreeAuthentication(BaseAuthentication):
    def authenticate(self, request):
        return True

def get_token_user(request):
    token = request.headers['authorization'].split(' ')[1]
    token_user = Token.objects.get(key = token).user
    return token_user
    
    
class OTPExistsPermission(BasePermission):
    def has_permission(self, request, view):
        try:
            otp = PasswordResetRequest.objects.get(
            code=request.data['otp'],
            user=User.objects.get(email=request.data['email'])
            )
            return True
        except:
            return False
    
class UserViewSet(viewsets.ModelViewSet):
    serializer_class = UserSerializer
    
    def get_permissions(self):
        if self.action == 'create':
            return []
        else:
            return [permissions.IsAuthenticated()]
        
    def get_queryset(self):
        return User.objects.all()
    
    @transaction.atomic
    def create(self, request):
        try:
            transfer_device_history = True if request.data['transfer_device_history'] == 'true' else False
        except:
            transfer_device_history = False
            
        try:
            identifier = request.data['identifier']
        except:
            identifier = ''
        serializer = self.get_serializer(data = request.data)
        if serializer.is_valid():
            new_user= User.objects.create_user(**serializer.validated_data)
            new_user.is_active =False
            x, _ = Group.objects.get_or_create(
                    name='Users'
                )
            
            new_user.groups.add(
                x.pk
            )
            new_user.save()
            
            for item in GuestScannedFile.objects.filter(identifier = identifier):
                if transfer_device_history == True:
                    newfile = ScannedFile.objects.create(
                        user=new_user,
                        file = item.file
                    )
                    
                    for page in item.guestscannedimage_set.all():
                        ScannedImage.objects.create(
                            file = newfile,
                            image = page.image
                        )
                item.delete()
                
            send_verification_email(new_user)
                
            return Response(serializer.data)
        else:
            print(serializer.errors)
        
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )
    
    @transaction.atomic
    @action(detail=True, methods=['POST'], url_path='reset-password')
    def reset_password(self, request, *args, **kwargs):
        self_obj = self.get_object()
        old_password = request.data['old_password']
        if self_obj.check_password(old_password):
            new_password = request.data['new_password']
            self_obj.set_password(new_password)
            self_obj.save()
            if self_obj.check_password(new_password):
                return Response(
                    UserSerializer(self_obj).data,
                    status = status.HTTP_200_OK
                )
            
            return Response(
                {'error':'Something went wrong.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        return Response(
            {'error':'Old password is incorrect.'},
            status=status.HTTP_400_BAD_REQUEST
        )
        
    @action(detail=False, methods=['GET'], url_path='get-user-info')
    def get_user_info(self, request, *args, **kwargs):
        token = self.request.headers['authorization'].split(' ')[1]
        token_user = Token.objects.get(key = token).user
        return Response(
            UserSerializer(token_user).data,
            status=status.HTTP_200_OK
        )
    
    def partial_update(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        # self.perform_update(serializer)

        return Response(serializer.data, status=status.HTTP_200_OK)

class FileInputViewSet(viewsets.ModelViewSet):
    serializer_class = FileInputSerializer
    pagination_class = PageNumberPagination
    permission_classes = [DjangoModelPermissions]
    
    def get_queryset(self):
        token = self.request.headers['authorization'].split(' ')[1]
        token_user = Token.objects.get(key = token).user
        return FileInput.objects.filter(user=token_user).order_by('-created')

    @transaction.atomic
    def create(self, request):
        token = self.request.headers['authorization'].split(' ')[1]
        token_user = Token.objects.get(key = token).user
        if isinstance(request.data, QueryDict):
            request.data._mutable = True
        request.data['user'] = token_user.pk
        serializer = FileInputSerializer(data = request.data)
        if serializer.is_valid():
            formitem = serializer.save()
            
            uploaded_file = formitem.file
            output_folder = "Output_images"
            convert_to_image_test(uploaded_file, output_folder, formitem, dpi=300, quality=90)   
            pages = [x for x in formitem.filepage_set.all()] 
            filename = formitem.file.name.split('/')[-1]
            
            response = {}
            response['file']=serializer.data
            return Response(
                response,
                status=status.HTTP_200_OK
            )
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )

class FilePageViewSet(viewsets.ModelViewSet):
    serializer_class = FilePageSerializer
    permission_classes = [DjangoModelPermissions]
    pagination_class = PageNumberPagination
    
    def get_queryset(self):
        token = self.request.headers['authorization'].split(' ')[1]
        token_user = Token.objects.get(key = token).user
        return FilePage.objects.filter(user=token_user).order_by('-created')

    @action(detail=True, methods=['POST'], url_path='convert-to-document')
    def convert_to_word(self, request, *args, **kwargs):
        file_obj = self.get_object()
        x = page_to_document(file_obj)
        
        response = HttpResponse(x, content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
        response['Content-Disposition'] = 'attachment; filename="example.docx"'
        
        return response

    
#table extraction 
class ImageInputViewSet(viewsets.ModelViewSet):
    serializer_class = ImageInputSerializer
    permission_classes = [DjangoModelPermissions]
    pagination_class = PageNumberPagination
    
    def get_queryset(self):
        token = self.request.headers['authorization'].split(' ')[1]
        token_user = Token.objects.get(key = token).user
        return ImageInput.objects.filter(user=token_user).order_by('-created')

    @transaction.atomic
    def create(self,request):
        data = request.data.copy()
        request.data._mutable = True
        token = self.request.headers['authorization'].split(' ')[1]
        token_user = Token.objects.get(key = token).user
        request.data['user'] = token_user.pk
        serializer = ImageInputSerializer(data=request.data)
        # serializer.is_valid()
        # if serializer.is_valid():
            #image = serializer.save()
        #     x=new_image_processing(image)
            
        #     if isinstance(x, list):
        #         to_save = []
        #         response = {}
        #         response['images']=serializer.data
        #         response['imagedata']=x
                
        #         for item in x:
        #             x_json = item.to_dict(orient='list')
        #             x_json = json.dumps(x_json)
        #             to_save.append(x_json) 
                    
        #         image.table = to_save
        #         image.save()
        #         return Response(
        #             response,
        #             status=status.HTTP_200_OK
        #         )
        #     image.delete()
        #     return Response(
        #             {'error': 'No tables found.'},
        #             status = status.HTTP_400_BAD_REQUEST
        #         )

        # return Response(
        #     serializer.errors,
        #     status=status.HTTP_400_BAD_REQUEST
        # )

        serializer.is_valid(raise_exception=True)
        image = serializer.save()
        try:
            tables_list = process_image_for_table_extraction(image.image.path)
            
            if tables_list:
                response = {
                    'images': serializer.data,
                    'imagedata': [df.to_dict(orient='list') for df in tables_list]
                }
                
                image.table = [json.dumps(df.to_dict(orient='list')) for df in tables_list]
                image.save()
                return Response(response, status=status.HTTP_200_OK)
            
            image.delete()
            return Response({'error': 'No tables found.'}, status=status.HTTP_400_BAD_REQUEST)
        
        except Exception as e:
            image.delete()
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
    @action(detail=False, methods=['POST'], url_path='table-from-pdf-page')
    def extract_table_from_pdf_page(self, request):
        page = FilePage.objects.get(id=request.data['id'])
        to_serialize = {
            'image': page.image,
            'user': get_token_user(request).pk
        }
        serializer = self.get_serializer(data=to_serialize)
        serializer.is_valid(raise_exception=True)
        
        image = serializer.save()
        try:
            tables_list = process_image_for_table_extraction(image.image.path)
            
            if tables_list:
                response = {
                    'images': serializer.data,
                    'imagedata': [df.to_dict(orient='list') for df in tables_list]
                }
                
                image.table = [json.dumps(df.to_dict(orient='list')) for df in tables_list]
                image.save()
                return Response(response, status=status.HTTP_200_OK)
            
            image.delete()
            return Response({'error': 'No tables found.'}, status=status.HTTP_400_BAD_REQUEST)
        
        except Exception as e:
            image.delete()
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # @action(detail=False, methods=['POST'],url_path='table-from-pdf-page')
    # def extract_table_from_pdf_page(self, request):
    #     page = FilePage.objects.get(id = request.data['id'])
    #     to_serialize = {
    #         'image':page.image,
    #         'user': get_token_user(request).pk
            
    #     }
    #     serializer = self.get_serializer(data=to_serialize)
    #     if serializer.is_valid():
    #         image = serializer.save()
    #         x=new_image_processing(image)
            
    #         if isinstance(x, list):
    #             to_save = []
    #             response = {}
    #             response['images']=serializer.data
    #             response['imagedata']=x
                
    #             for item in x:
    #                 x_json = item.to_dict(orient='list')
    #                 x_json = json.dumps(x_json)
    #                 to_save.append(x_json) 
                    
    #             image.table = to_save
    #             image.save()
    #             return Response(
    #                 response,
    #                 status=status.HTTP_200_OK
    #             )
                
    #         return Response(
    #                 {'error': 'No tables found.'},
    #                 status = status.HTTP_400_BAD_REQUEST
    #             )

    #     return Response(
    #         serializer.errors,
    #         status=status.HTTP_400_BAD_REQUEST
    #     )
        
#pdf conversion
class ScannedFileViewSet(viewsets.ModelViewSet):
    pagination_class = PageNumberPagination
    serializer_class = ScannedFileSerializer
    permission_classes = [DjangoModelPermissions]
    
    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['request'] = self.request  
        return context
    
    def get_queryset(self):
        token = self.request.headers['authorization'].split(' ')[1]
        token_user = Token.objects.get(key = token).user
        return ScannedFile.objects.filter(user=token_user).order_by('-created')
    
    @transaction.atomic
    def create(self, request):
        if isinstance(request.data, QueryDict):
            request.data._mutable = True
        token = request.headers['authorization'].split(' ')[1]
        token_user = Token.objects.get(key = token).user
        request.data.update(
            {'user': token_user.pk}
        )
        # xdata['user'] = token_user.pk
        serializer = ScannedFileSerializer(data=request.data)
        if serializer.is_valid():
            file = serializer.save()
            try:
                images = request.data.getlist('images')
            except:
                images = [request.data['images']]
            imagelist = []
            for item in images:
                filename=item.name.split('.')[0]
                # y=camscanner_effect(item)
                y = new_process_image(item)
                _, img_encoded = cv2.imencode('.png', y)
                image_io = BytesIO(img_encoded)
                content_file = ContentFile(image_io.getvalue())
                
                img = ScannedImage.objects.create(
                    file = file,
                )
                img.image.save(f'{filename}.png',content_file)
                img.save()
                imagelist.append(img.image.path)
            
            pdf = create_pdf(imagelist)
            current_datetime = datetime.now()
            formatted_datetime = f"{current_datetime:%Y-%m-%d %H:%M:%S}"
            
            file.file.save(f'{token_user.username}-{formatted_datetime}.pdf',ContentFile(pdf))

            
            return Response(
                serializer.data
            )
        else:
            return Response({
                'errors':serializer.errors
            })

class ScannedImageViewSet(viewsets.ModelViewSet):
    pagination_class = PageNumberPagination
    permission_classes = [DjangoModelPermissions]
    serializer_class = ScannedImageSerializer
    
    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['request'] = self.request  # Assuming self.request is the request object
        return context
    
    def get_queryset(self):
        token = self.request.headers['authorization'].split(' ')[1]
        token_user = Token.objects.get(key = token).user
        return ScannedImage.objects.filter(user=token_user).order_by('-created')
    
    
    # def list(self, request, *args, **kwargs):
    #     serializer_context = {'request': request}
    #     serializer = self.serializer_class(self.queryset, context=serializer_context, many=True)
    #     return Response(serializer.data)

    # def retrieve(self, request, *args, **kwargs):
    #     instance = self.get_object()
    #     serializer_context = {'request': request}
    #     serializer = self.serializer_class(instance, context=serializer_context)
    #     return Response(serializer.data)

class WordConversionViewSet(viewsets.ModelViewSet):
    pagination_class = PageNumberPagination
    serializer_class = WordConversionSerializer
    
    def get_queryset(self):
        token = self.request.headers['authorization'].split(' ')[1]
        token_user = Token.objects.get(key = token).user
        return WordConversion.objects.filter(user=token_user).order_by('-created')
    
    # def create(self, request):
    #     file_obj = request.data['image']
    #     page_obj = FilePage.objects.create(
    #         image = file_obj
    #     )
    #     x = page_to_document(page_obj)
        
    #     token = request.headers['authorization'].split(' ')[1]
    #     token_user = Token.objects.get(key = token).user
    #     current_datetime = datetime.now()
    #     formatted_datetime = f"{current_datetime:%Y-%m-%d %H:%M:%S}"
    #     file_name = f'{token_user.username}-{formatted_datetime}'
    #     x = x.getvalue()
    #     document_model = WordConversion.objects.create(
    #         user = token_user,
    #         document = ContentFile(x, name=f'{file_name}.docx'),
    #         image = page_obj.image
    #     )
    #     page_obj.delete()
        
    #     return Response(
    #         WordConversionSerializer(document_model).data,
    #         status = status.HTTP_200_OK
    #     )
        
    # @action(detail=False, methods=['POST'], url_path='word-from-pdf-page')
    # def word_from_pdf_page(self, request):
    #     page_obj = FilePage.objects.get(id = request.data['id'])
    #     x = page_to_document(page_obj)
        
    #     token = request.headers['authorization'].split(' ')[1]
    #     token_user = Token.objects.get(key = token).user
    #     current_datetime = datetime.now()
    #     formatted_datetime = f"{current_datetime:%Y-%m-%d %H:%M:%S}"
    #     file_name = f'{token_user.username}-{formatted_datetime}'
    #     x = x.getvalue()
    #     document_model = WordConversion.objects.create(
    #         user = token_user,
    #         document = ContentFile(x, name=f'{file_name}.docx'),
    #         image = page_obj.image
    #     )
    #     page_obj.delete()
        
    #     return Response(
    #         WordConversionSerializer(document_model).data,
    #         status = status.HTTP_200_OK
    #     )
        
        
    #     pass
    def create(self, request):
        file_obj = request.data['image']
        page_obj = FilePage.objects.create(image=file_obj)
        
        doc_buffer = convert_image_to_docx(page_obj.image.path)
        
        token = request.headers['authorization'].split(' ')[1]
        token_user = Token.objects.get(key=token).user
        current_datetime = django_now()
        formatted_datetime = f"{current_datetime:%Y-%m-%d %H:%M:%S}"
        file_name = f'{token_user.username}-{formatted_datetime}'
        doc_buffer.seek(0)
        
        document_model = WordConversion.objects.create(
            user=token_user,
            document=ContentFile(doc_buffer.read(), name=f'{file_name}.docx'),
            image=page_obj.image
        )
        page_obj.delete()
        
        return Response(
            WordConversionSerializer(document_model).data,
            status=status.HTTP_200_OK
        )
        
    @action(detail=False, methods=['POST'], url_path='word-from-pdf-page')
    def word_from_pdf_page(self, request):
        page_obj = FilePage.objects.get(id=request.data['id'])
        
        doc_buffer = convert_image_to_docx(page_obj.image.path)
        
        token = request.headers['authorization'].split(' ')[1]
        token_user = Token.objects.get(key=token).user
        current_datetime = django_now()
        formatted_datetime = f"{current_datetime:%Y-%m-%d %H:%M:%S}"
        file_name = f'{token_user.username}-{formatted_datetime}'
        doc_buffer.seek(0)
        
        document_model = WordConversion.objects.create(
            user=token_user,
            document=ContentFile(doc_buffer.read(), name=f'{file_name}.docx'),
            image=page_obj.image
        )
        page_obj.delete()
        
        return Response(
            WordConversionSerializer(document_model).data,
            status=status.HTTP_200_OK
        )
        
    
class GuestScannedFileViewSet(viewsets.ModelViewSet):
    pagination_class = PageNumberPagination
    serializer_class = GuestScannedFileSerializer
    # authorization_classes = [PostOnlyFreeAuthentication]
    permission_classes = [AllowAny]
    
    def get_queryset(self):
        if self.request.COOKIES.get('identifier'):
            print('cookie:', self.request.COOKIES.get('identifier'))
            return GuestScannedFile.objects.filter(identifier = self.request.COOKIES.get('identifier')).order_by('-created')
        return GuestScannedFile.objects.none()
    
    @transaction.atomic
    def create(self, request):
        request.data._mutable = True
        data = request.data
        if request.COOKIES.get('identifier'):
            identifier = request.COOKIES.get('identifier')
        else:
            identifier = str(uuid.uuid4())
        data['identifier'] = identifier
        serializer = self.get_serializer(data = data)
        if serializer.is_valid():
            file = serializer.save()
            try:
                images = data.getlist('images')
            except:
                images = [data['images']]
                
            imagelist = []
            for item in images:
                filename=item.name.split('.')[0]
                # y=camscanner_effect(item)
                y=process_image(item)
                _, img_encoded = cv2.imencode('.png', y)
                image_io = BytesIO(img_encoded)
                content_file = ContentFile(image_io.getvalue())
                
                img = GuestScannedImage.objects.create(
                    file = file,
                )
                img.image.save(f'{filename}.png',content_file)
                img.save()
                imagelist.append(img.image.path)
            
            pdf = create_pdf(imagelist)
            current_datetime = datetime.now()
            formatted_datetime = f"{current_datetime:%Y-%m-%d %H:%M:%S}"
            
            file.file.save(f'{identifier}-{formatted_datetime}.pdf',ContentFile(pdf))
            
            response = Response(GuestScannedFileOutputSerializer(file).data)
            response.set_cookie(
                'identifier',
                identifier,
                httponly=True
            )
    
            return response
        else:
            return Response({
                'errors':serializer.errors
            })

class BugImageViewSet(viewsets.ModelViewSet):
    serializer_class = BugImageSerializer
    
    def get_queryset(self):
        return BugImage.objects.all()
    
class BugReportViewSet(viewsets.ModelViewSet):
    serializer_class = BugReportSerializer
    
    def get_queryset(self):
        return BugReport.objects.all()
    
    def create(self, request):
        serializer = self.get_serializer(data = request.data)
        if serializer.is_valid():
            report_obj = serializer.save()
            
            try:
                images = request.data.getlist('images')
            except:
                images = [request.data['images']]
                
            for item in images:
                BugImage.objects.create(
                    report=report_obj,
                    image=item
                )
            
            return Response(
                serializer.data,
                status = status.HTTP_200_OK
            )
            
        return Response(
            serializer.errors,
            status = status.HTTP_400_BAD_REQUEST
        )

class ContactFormViewSet(viewsets.ModelViewSet):
    serializer_class = ContactFormSerializer
    permission_classes = [AllowAny]
    
    def get_queryset(self):
        return ContactForm.objects.all()
    

class PasswordResetRequestViewSet(viewsets.ModelViewSet):
    pagination_class = PageNumberPagination
    serializer_class = PasswordResetRequestSerializer
    permission_classes = [AllowAny]
    
    def get_queryset(self):
        return PasswordResetRequest.objects.all()
    
    def create(self, request):
        try:
            user = User.objects.get(email = request.data['email'])
        except User.DoesNotExist:
            return Response(
                {'error':'User not found.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        requests = PasswordResetRequest.objects.filter(user = user).order_by('-created')
        if len(requests) >0:
            now = timezone.now()
            threshold = now - timedelta(minutes=5)
            if (requests[0].created >= threshold and requests[0].is_active==True):
                return Response(
                    {
                        'error':'There is already an existing password reset request created within the last 5 mins. Please check your email.'
                    },
                    status=status.HTTP_404_NOT_FOUND
                )
        for item in requests:
            item.is_active=False
            item.save()
        
        if isinstance(request.data, QueryDict):
            request.data._mutable = True
        request.data['user']=user.pk
        serializer = self.get_serializer(data=request.data)
        
        if serializer.is_valid():
            request_obj = serializer.save()
            send_password_reset_email(request_obj)
            
            return Response(
                serializer.data,
                status=status.HTTP_200_OK            
            )
        return Response(
            status=status.HTTP_400_BAD_REQUEST
        )

class CustomAuthenticationSerializer(AuthTokenSerializer):
    
    def validate(self, attrs):
        username = attrs.get('username')
        password = attrs.get('password')

        if username and password:
            user = User.objects.get(username=username)
            if not user.is_active:
                raise serializers.ValidationError(
                    'User is not yet verified.',
                    code='authorization'
                )
            user = authenticate(request=self.context.get('request'),
                                username=username, password=password)

            # The authenticate call simply returns None for is_active=False
            # users. (Assuming the default ModelBackend authentication
            # backend.)
            if not user:
                msg = _('Unable to log in with provided credentials.')
                raise serializers.ValidationError(msg, code='authorization')
        else:
            msg = _('Must include "username" and "password".')
            raise serializers.ValidationError(msg, code='authorization')

        attrs['user'] = user
        return attrs
    
class ObtainExpiringAuthToken(ObtainAuthToken):
    def post(self, request):
        serializer = CustomAuthenticationSerializer(data=request.data)
        if serializer.is_valid():     
            token, created =  Token.objects.get_or_create(user=serializer.validated_data['user'])

            if not created and token.created < timezone.now() - timedelta(hours=24):
                token.delete()
                token = Token.objects.create(user=serializer.validated_data['user'])
                token.created = datetime.now()
                token.save()
            token_user = token.user
            response = UserSerializer(token_user).data
            response['token'] = token.key
            
            return Response(response)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

obtain_expiring_auth_token = ObtainExpiringAuthToken.as_view()

@api_view(['GET'])
def get_table_from_pdf_page(request):
    item = FilePage.objects.get(id=request.query_params['id'])
    
    x= new_image_processing(item)
    response = {}
        
    response['page']=FilePageSerializer(item).data
    response['imagedata']=x
    return Response(
        response,
        status=status.HTTP_200_OK
    )
    
def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image_stream = BytesIO(image_data)
    image = Image.open(image_stream)
    
    return image

def save_image(image, filename):
    image.save(filename)

@api_view(['POST'])
def test_api(request):
    return Response(
        {'a':'b'}
    )

@api_view(['GET'])
@permission_classes([AllowAny])
@authentication_classes([])
def send_email_test(request):
    html_message = render_to_string(
        'verification_email.html', 
        {
            'user': 'new_user',
            
        }
    )
    plain_message = strip_tags(html_message)
        
    send_mail(
        subject = 'User Verification',
        message = plain_message,
        from_email = settings.DEFAULT_FROM_EMAIL,
        recipient_list=['test.ran.mail@gmail.com'],
        fail_silently=False,
        html_message=html_message
        )
    
    return Response({'a':'b'})


def send_verification_email(user:User):
    for item in VerificationEmail.objects.filter(user = user):
        item.status = False
        item.save()
        
    e = VerificationEmail.objects.create(
        user = user
    )
    print(e.id)
    
    html_message = render_to_string(
        'verification_email.html',
        {
            'user':user,
            'email_obj': e,
            'ip':get_server_ip()
        }
    )
    plain_message = strip_tags(html_message)
    
    send_mail(
        subject = 'User Verification',
        message = plain_message,
        from_email = settings.DEFAULT_FROM_EMAIL,
        recipient_list=[user.email],
        fail_silently=False,
        html_message=html_message
    )

def send_password_reset_email(resetRequest:PasswordResetRequest):
    
    html_message = render_to_string(
        'password_reset_email.html',
        {
            'resetRequest':resetRequest,
            'ip':get_server_ip()
        }
    )
    plain_message = strip_tags(html_message)
    
    send_mail(
        subject = 'Password Reset',
        message = plain_message,
        from_email = settings.DEFAULT_FROM_EMAIL,
        recipient_list=[resetRequest.user.email],
        fail_silently=False,
        html_message=html_message
    )
    
    
@api_view(['GET'])
@permission_classes([AllowAny])
def verify_user(request):
    id = request.GET['id']
    obj = VerificationEmail.objects.get(pk= id)
    user = obj.user
    
    if user.is_active == True:
        return redirect('verification-failed')
    
    if timezone.now() > obj.created_on + timedelta(hours=24):
        obj.status = False
        obj.save()
        return redirect('verification-expired')
    obj.status = True
    obj.save()
    user.is_active=True
    user.save()
    
    return redirect('verification-success')

@api_view(['POST'])
@permission_classes([AllowAny])
def check_password_otp(request):
    code = request.data['otp']
    user = User.objects.get(email = request.data['email'])
    try:
        request_obj = PasswordResetRequest.objects.get(
        code=code,
        user = user
       )
        now = timezone.now()
        threshold = now - timedelta(minutes=5)
        if (now - request_obj.created <= timedelta(minutes=5) and request_obj.is_active==True):
            return Response(
            PasswordResetRequestSerializer(request_obj).data,
            status=status.HTTP_200_OK
            )
        return Response(
            {'error':'OTP timed out. Press resend to try again.'},
            status=status.HTTP_400_BAD_REQUEST
            
        )

    except PasswordResetRequest.DoesNotExist:
        return Response(
            {
                'error': 'Invalid OTP.'
            },
            status=status.HTTP_400_BAD_REQUEST
        )

@api_view(['POST'])
@permission_classes([OTPExistsPermission])
def change_password(request):
    reset_obj = PasswordResetRequest.objects.get(code=request.data['otp'])
    if reset_obj.is_active == True:
        user=reset_obj.user
        user.set_password(request.data['password'])
        user.save()
        user.check_password(request.data['password'])
        reset_obj.is_active=False
        reset_obj.save()
        return Response(
            UserSerializer(user).data,
            status=status.HTTP_200_OK
        )
    else:
        return Response(
            {'error': 'Something went wrong.'},
            status = status.HTTP_400_BAD_REQUEST
        )


@api_view(['POST'])
def resend_verification_email(request):
    token = request.headers['authorization'].split(' ')[1]
    token_user = Token.objects.get(key = token).user
    send_verification_email(token_user)
    
    return Response(
        {
            'message':'Verification email sent.'
        },
        status = status.HTTP_200_OK
    )

def verify_success(request):
    return render(request, 'verification_success.html')

def verify_failed(request):
    return render(request, 'verification_failed.html')

def verify_expired(request):
    return render(request, 'verification_link_expired.html')

def get_server_ip():
    try:
        # Get the hostname of the server
        hostname = socket.gethostname()
        # Get the IP address associated with the hostname
        server_ip = socket.gethostbyname(hostname)
        return server_ip
    except socket.error as err:
        # Handle the error if any
        return None
    
@csrf_exempt
@api_view(['GET'])
@permission_classes([AllowAny])
def get_csrf_token(request):
    csrf_token = get_token(request)
    return Response({'csrf_token': csrf_token})

@api_view(['POST'])
def convert_to_word(request):
    file_obj = request.data['image']
    page_obj = FilePage.objects.create(
        image = file_obj
    )
    x = page_to_document(page_obj)
    
    token = request.headers['authorization'].split(' ')[1]
    token_user = Token.objects.get(key = token).user
    document_model = WordConversion.objects.create(
        user = token_user,
        file = ContentFile(x),
        image = file_obj 
    )
    
    response = HttpResponse(x, content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
    response['Content-Disposition'] = 'attachment; filename="example.docx"'
    
    return response