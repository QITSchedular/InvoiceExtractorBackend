from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from ..models import QitInvoicetemplate
from ..serializers import TemplateSerializer

@api_view(['POST'])
def create_template(request):
    if request.method == 'POST':
        serializer = TemplateSerializer(data=request.data)
        
        if serializer.is_valid():
            serializer.save()  # Save the data to the Template model
            return Response(serializer.data, status=status.HTTP_201_CREATED)  # Return created data
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)  # Return validation errors

# @api_view(['GET'])
# def get_templates(request):
#     if request.method == 'GET':
#         templates = QitInvoicetemplate.objects.all()  # Get all Template records
#         serializer = TemplateSerializer(templates, many=True)
#         return Response(serializer.data)

@api_view(['GET'])
def get_templates(request):
    if request.method == 'GET':
        templates = QitInvoicetemplate.objects.all()  # Fetch all templates
        serializer = TemplateSerializer(templates, many=True)

        # Transforming the data to the desired format
        formatted_data = {}
        for template in serializer.data:
            template_name = template['value'].replace(" ", "_").lower()  # Generate a key using the template value
            formatted_data[template_name] = {
                "keys": template['template']['keys'],
                "expected_values": template['template']['expected_values'],
                "table_structure": template['template']['table_structure']
            }
        
        return Response(formatted_data)