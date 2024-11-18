from rest_framework import serializers
from .models import QitInvoicetemplate

class FileUploadSerializer(serializers.Serializer):
    file = serializers.FileField()


class TemplateSerializer(serializers.ModelSerializer):
    class Meta:
        model = QitInvoicetemplate
        fields = ['value', 'template']  # Exclude transId as it's