from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .invoice import process_invoice_with_all_templates
import os

from ..models import QitInvoicetemplate
from ..serializers import TemplateSerializer
import tempfile
# @api_view(['POST'])
# def FileUploadView(request):
#     print("Content-Type:", request.META.get('CONTENT_TYPE'))
#     print("Files received:", request.FILES)
    
#     if 'file' not in request.FILES:
#         return Response({"error": "No file provided."}, status=status.HTTP_400_BAD_REQUEST)

#     file = request.FILES['file']
#     return Response({"message": "File uploaded successfully!", "filename": file.name}, status=status.HTTP_201_CREATED)

# @api_view(['POST'])
# def FileUploadView(request):
#     # Print Content-Type and files for debugging purposes
#     # print("Content-Type:", request.META.get('CONTENT_TYPE'))
#     print("Files received:", request.FILES)
    
#     # Check if the 'file' is present in the request
#     if 'file' not in request.FILES:
#         return Response({"error": "No file provided."}, status=status.HTTP_400_BAD_REQUEST)

#     # Retrieve the file from the request
#     file = request.FILES['file']
#     output_dir = r'./'
#     all_template_results = process_invoice_with_all_templates(file, output_dir)
#     print("all_template_results : ",all_template_results)
#     # Check if the uploaded file is a PDF (by checking the extension or MIME type)
#     if not file.name.endswith('.pdf'):
#         return Response({"error": "Only PDF files are allowed."}, status=status.HTTP_400_BAD_REQUEST)
    
#     # Optional: Check MIME type (Content-Type) for PDF
#     if file.content_type != 'application/pdf':
#         return Response({"error": "Invalid file type. Only PDF files are allowed."}, status=status.HTTP_400_BAD_REQUEST)

#     # If the file is a PDF, proceed with uploading
#     return Response({"message": "File uploaded successfully!", "filename": file.name}, status=status.HTTP_201_CREATED)




# @api_view(['POST'])
# def FileUploadView(request):
#     print("Files received:", request.FILES)

#     # Check if the 'file' is present in the request
#     if 'file' not in request.FILES:
#         return Response({"error": "No file provided."}, status=status.HTTP_400_BAD_REQUEST)

#     file = request.FILES['file']

#     # Check if the uploaded file is a PDF
#     if not file.name.endswith('.pdf') or file.content_type != 'application/pdf':
#         return Response({"error": "Only PDF files are allowed."}, status=status.HTTP_400_BAD_REQUEST)

#     # Save the file to a temporary directory
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#         for chunk in file.chunks():
#             temp_file.write(chunk)
#         temp_file_path = temp_file.name

#     try:
#         # Process the saved PDF file
#         output_dir = './'
#         all_template_results = process_invoice_with_all_templates(temp_file_path, output_dir)
#         # print("all_template_results:", all_template_results)
#     except Exception as e:
#         return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#     finally:
#         # Clean up the temporary file
#         os.remove(temp_file_path)

#     return Response({"message": "File uploaded and processed successfully!","all_template_results":all_template_results}, status=status.HTTP_201_CREATED)
import json

@api_view(['POST'])
def FileUploadView(request):
    print("Files received:", request.FILES)

    # Check if the 'file' is present in the request
    if 'file' not in request.FILES:
        return Response({"error": "No file provided."}, status=status.HTTP_400_BAD_REQUEST)

    file = request.FILES['file']

    # Check if the uploaded file is a PDF
    if not file.name.endswith('.pdf') or file.content_type != 'application/pdf':
        return Response({"error": "Only PDF files are allowed."}, status=status.HTTP_400_BAD_REQUEST)

    # Save the file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        for chunk in file.chunks():
            temp_file.write(chunk)
        temp_file_path = temp_file.name

    try:
        # Process the saved PDF file to extract data
        output_dir = './'
        templates = QitInvoicetemplate.objects.all()  # Get all Template records
        serializer = TemplateSerializer(templates, many=True)

        formatted_data = {}
        for template in serializer.data:
            template_name = template['value'].replace(" ", "_").lower()
            template_content = template['template']

            if isinstance(template_content, dict):
                formatted_data[template_name] = {
                    "keys": template_content.get('keys', {}),
                    "expected_values": template_content.get('expected_values', {}),
                    "table_structure": template_content.get('table_structure', [])
                }
            else:
                print(f"Error: Template content for {template_name} is not a dictionary.")
        # formatted_data = {}
        # for template in serializer.data:
        #     template_name = template['value'].replace(" ", "_").lower()  # Generate a key using the template value
        #     formatted_data[template_name] = {
        #         "keys": template['template']['keys'],
        #         "expected_values": template['template']['expected_values'],
        #         "table_structure": template['template']['table_structure']
        #     }
            # print("Serialized template data:", serializer.data)

        # print("=========",formatted_data)
        all_template_results = process_invoice_with_all_templates(temp_file_path, output_dir,formatted_data)
        # print("=========1",all_template_results)
        # print("=========2",output_dir)
        
        # Identify the template with the highest accuracy
        best_template = None
        max_accuracy = -1

        # print("=========3",all_template_results.items())
        for template_name, template_data in all_template_results.items():
            accuracy = template_data.get('accuracy', 0)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_template = {"template": template_data}
            # print("------1234----",best_template)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)
    if best_template:
        return Response({"message": "File uploaded and processed successfully!", "best_template": best_template}, status=status.HTTP_201_CREATED)
    else:
        return Response({"message": "No valid templates found."}, status=status.HTTP_204_NO_CONTENT)
