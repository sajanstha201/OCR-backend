"""
URL configuration for table_extraction project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from rest_framework import routers, serializers, viewsets
from file_to_table.api.viewsets import *
from file_to_table.views import show_homepage
from django.conf.urls.static import static

router = routers.DefaultRouter()
router.register(r'users', UserViewSet, basename='users')
router.register(r'files', FileInputViewSet, 'files')
router.register(r'file-page', FilePageViewSet, 'file-page')
router.register(r'images', ImageInputViewSet, 'image')
router.register(r'scanned-files', ScannedFileViewSet, 'scanned-files')
router.register(r'scanned-images', ScannedImageViewSet, 'scanned-images')
router.register(r'guest-scanned-files', GuestScannedFileViewSet, 'guest-scanned-files')
router.register(r'convert-doc', WordConversionViewSet, 'word-conversion')
router.register(r'forgot-password', PasswordResetRequestViewSet, 'forgot-password')
router.register(r'contact-form', ContactFormViewSet, 'contact-us-form')
router.register(r'bug-report', BugReportViewSet, 'bug-report')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api-auth/', include('rest_framework.urls')),
    path('api/', include(router.urls)),
    path('api/get-table-from-pdf-page/', get_table_from_pdf_page),
    path('', show_homepage, name='show_homepage'),
    path('api/login/', obtain_expiring_auth_token),
    # path('api/test/', get_server_ip),
    path('api/VerifyEmail', verify_user),
    path('api/resend-verification-email', resend_verification_email),
    # path('verification-failed', verify_failed),
    # path('verification-success', verify_success),
    # path('verification-expired', verify_expired),
    path('', include('file_to_table.urls')),
    path('api/get-csrf-token/', get_csrf_token),
    # path('api/convert-doc/',convert_to_word)
    path('api/check-otp/', check_password_otp),
    path('api/change-password/', change_password),

]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)