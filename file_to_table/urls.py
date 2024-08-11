from django.urls import path, include
from .api.viewsets import *

urlpatterns = [
    path('verification-failed', verify_failed, name='verification-failed'),
    path('verification-success', verify_success, name='verification-success'),
    path('verification-expired', verify_expired, name='verification-expired'),

]
