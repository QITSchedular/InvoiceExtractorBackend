from django.urls import path
from .Views.PDF_Upload import FileUploadView  # Importing from the Views folder
from .Views.ExtractTextView import ExtractTextView  # Importing from the Views folder
from .Views.Add_template import create_template,get_templates  # Importing from the Views folder
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('uploadFile/', FileUploadView),
    path('addTemplate/', create_template),
    path('getTemplate/', get_templates),
    path('extractText/', ExtractTextView),
]
