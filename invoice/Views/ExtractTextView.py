# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from rest_framework.parsers import JSONParser
# from rest_framework import status
# import fitz  # PyMuPDF library
# import os

# # Define the path to the static PDF file
# PDF_FILE_PATH = "api/invoice_8 1.pdf"

# @csrf_exempt
# def extract_text_view(request):
#     if request.method != 'POST':
#         return JsonResponse({'error': 'Only POST method is allowed.'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

#     # Parse JSON data from the request
#     data = JSONParser().parse(request)
#     page_number = int(data.get('page', 0))
#     x1 = data.get('x1')
#     y1 = data.get('y1')
#     x2 = data.get('x2')
#     y2 = data.get('y2')

#     # Validate input coordinates
#     if not all([x1, y1, x2, y2]):
#         return JsonResponse({'error': 'Coordinates are required.'}, status=status.HTTP_400_BAD_REQUEST)

#     try:
#         # Convert coordinates to float
#         x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))

#         # Check if the static PDF file exists
#         if not os.path.exists(PDF_FILE_PATH):
#             return JsonResponse({'error': 'Static PDF file not found.'}, status=status.HTTP_404_NOT_FOUND)

#         # Open the PDF file and extract text
#         with fitz.open(PDF_FILE_PATH) as pdf:
#             # Validate page number
#             if page_number < 0 or page_number >= pdf.page_count:
#                 return JsonResponse({'error': 'Page number out of range.'}, status=status.HTTP_400_BAD_REQUEST)

#             # Extract text within the given rectangle
#             page = pdf[page_number]
#             rect = fitz.Rect(x1, y1, x2, y2)
#             extracted_text = page.get_text("text", clip=rect)

#         return JsonResponse({'text': extracted_text}, status=status.HTTP_200_OK)

#     except Exception as e:
#         return JsonResponse({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


import fitz  # PyMuPDF
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view, parser_classes

@api_view(['POST'])
@parser_classes([MultiPartParser])
def ExtractTextView(request):
    pdf_file = request.FILES.get('file')
    print("pdf_file : ",pdf_file)
    
    page_number = int(request.data.get('page', 0))
    x1 = request.data.get('x1')
    x2 = request.data.get('x2')
    y1 = request.data.get('y1')
    y2 = request.data.get('y2')

    # Validate inputs
    # if not pdf_file or not x1 or not x2 or not y1 or not y2:
    #     return Response({'error': 'PDF file and coordinates are required.'}, status=status.HTTP_400_BAD_REQUEST)
    # Validate inputs
    if not pdf_file:
        return Response({'error': 'PDF file is required.'}, status=status.HTTP_400_BAD_REQUEST)

    # Check if coordinates are valid (converted to float or int if needed)
    try:
        x1 = float(x1) if x1 else None
        x2 = float(x2) if x2 else None
        y1 = float(y1) if y1 else None
        y2 = float(y2) if y2 else None
    except ValueError:
        return Response({'error': 'Invalid coordinate values.'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # Parse coordinates
        x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))

        # Load PDF and extract text
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as pdf:
            # Check if the page number is within bounds
            if page_number < 0 or page_number >= pdf.page_count:
                return Response({'error': 'Page number out of range.'}, status=status.HTTP_400_BAD_REQUEST)

            page = pdf[page_number]
            rect = fitz.Rect(x1, y1, x2, y2)
            extracted_text = page.get_text("text", clip=rect)

        return Response({'text': extracted_text}, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    