import re
import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import json
import numpy as np
import cv2
import fitz
import pickle

# Path to Tesseract OCR executable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract' 

def preprocess_image(image):
    # Convert PIL image to OpenCV format
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    # Convert to grayscale
    img_gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    img_blur = cv2.GaussianBlur(img_gray, (5 , 5), 0)

    # Adaptive Thresholding (useful for varying lighting conditions)
    img_thresh = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 12, 3)

    # Optional: Erosion and dilation to enhance text regions
    kernel = np.ones((3, 3), np.uint8)
    img_dilated = cv2.dilate(img_thresh, kernel, iterations=1)
    img_eroded = cv2.erode(img_dilated, kernel, iterations=1)

    # Convert back to PIL image
    preprocessed_img = Image.fromarray(img_eroded)
    return preprocessed_img

from spellchecker import SpellChecker

def correct_spelling(text):
    spell = SpellChecker()
    words = text.split()
    corrected_text = " ".join([spell.correction(word) for word in words])
    return corrected_text

# Custom corrections for common OCR mistakes
def correct_common_ocr_mistakes(text):
    text = text.replace('0', 'O')  
    text = text.replace('1', 'I')
    return text

data_model = {
    'invoice_number': ["Invoice No", "Invoice Number", "Invoice #","Inv No."],
    'invoice_date': ["Invoice Date"],
    'vendor_name': ["Vendor Name", "VENDOR NAME", "Customer Name"], 
    'GST NO': ["GST No", "GST NO","GSTIN Number","GSTIN Numbe","GSTTIN/UIN"],
    'PAN No': ["PAN No", "PAN Number"],
    'StateCode': ['State Code'],
    'Contact': ["Contact No", "Phone", "Phone No"],
    'Vendor Contact': ['VENDOR CONTACT NO'],
    'E-mail': ["Email", 'E-mail ID'],
    'Total Amount': ["Total Amount", "Grand Total", "Total"],
    'After Tax Amount': ["After Tax", "Amount After Tax"],
    'Website': ["Website"],
    'place_of_supply': ["Place of Supply", "Supply Place","Place Of Suppy"],
    'CIN NO': ["CIN No"],
    'Currency': ["Currency"],
    'Vehicle No.':["Vehicle No","Vehicle No.","Vehicle Number"],
    'Agent Code': ["Agent Code"],
    'IRN No.': ["IRN No.", "IRN No"],
    'Payment Terms': ["Payment Terms"],
    'Acknowledgement No.': ["Acknowledgement No","Acknowledge Number"],
    'Acknowledgement Date':["Acknowledge Date"],
    'Transporter': ["Transporter"],
    'Transportation Mode': ["Transportation Mode"],
    'Factory address': ["Factory Address", 'Factory Add'],
    'Office address': ["Office Address", 'Office Add',"Office"],
    'Vendor Address': ["Vendor Address"],
    'LR No': ['LR No', 'LR no', 'LR No.'],
    'LR Date': ['LR Date', 'LR date'],
    'Party P.O. Ref': ['Party P.O. Ref', 'Party P.O. ref'],
    'Purchase Order': ['PURCHASE ORDER'],
    'Purchase PO Date': ['PURCHASE PO DATE','Purchase PO Date'],
    'Declaration': ['Declaration', 'declaration'],
    'Destination': ['Destination', 'Final Destination'],
    'Booking': ['Booking', 'booking'],
    'Bill Date':['eWay Bill Date','eWay Bill Valid Date'],
    'Machine Number':['Machine No.','Machine Number','Machine no.','Machine No'],
    'Date Of Supply':['Date Of Supply'],
    'Date':["Date"],
    'State':['State'],
    "Party Bill No":["Party Bill No"],
    "Eway Bill No.":["Eway Bill No."],
    "Removal Date Time":["Removal Date Time"],
    "Challan No":["Challan No"],
    "Document":["DOCUMENT"],
    "Document Date":["DOCUMENT Date"],
    "Messers":["Messers"],
    "State":["State"],
    "GSTIIN/UIN":["GSTTIN/UIN"],
    "BILL NO.":["BILL NO"],
    "Issue Date":["DATE OF ISSUE"],
    "Job No.":["JOB NO"],
    "Job Date":["JOB Date"],
    "number_of_packs":["No. of Pack"],
    "Gr. Wt.":["Gr. Wt."],
    "Vessel":["VESSEL"],
    "Cif value":["CIF Value Rs"],
    "Port of Ship":["Port of Ship"],
    "BL No.":["BL No"],
    "Inv No.":["Inv No"],
    "B/E No":["B/E No"],
    "IGM Item No":["IGM ITEM No"],
    "Port of Del":["Port of Del"],
    "Description":["Description"],
    "Regd Office":["Regd Office"],
    "Pune Office":["Pune Office"],
    "Manglore":["Manglore"],
    "Receiver Details":['Receiver Details Billed to' , 'Receiver Details','Billed to'],
    "Duplicate for Supplier":['Duplicate for Supplier','Extra Copy']
}

# templates = {
    # "template_1": {
    #     "keys": {
    #         "GSTIN Number": ["GST No", "GSTIN", "GST NO","GSTIN Number","GSTTIN/UIN"],
    #         "PAN Number": ["PAN Number","PAN No"],
    #         "CIN NO": ["CIN NO","CIN","CIN No"],
    #         "invoice_number": ["Invoice No", "Invoice #", "Invoice Number"],
    #         "invoice_date": ["Invoice Date","Invoice date"],
    #         "Party P.O. Ref": ["Party P.O. Ref"],
    #         "Date": ["Date"],
    #         "Date of Supply": ["Date of Supply","Date Of Supply"],
    #         "Transportation Mode": ["Transportation Mode"],
    #         "Vehicle No": ["Vehicle No","Vehicle Number","Vehicle NO."],
    #         "Transporter Name": ["Transporter Name"],
    #         "LR No": ["LR No"],
    #         "LR Date": ["LR Date"]
    #     },
    #     "table_structure": {
    #         "columns": ["SR. No.", "Name of Product/Service", "Product Code", "HSN/SAC Code", "QTY.", "UOM", "Rate", "Amount", "Discount", "Taxable Value", "Value"],
    #     },
    #     "expected_values": {
    #         "GSTIN Number": "24AAYCS6904J1ZQ",
    #         "PAN Number": "AAYCS6904J",
    #         "CIN NO": "L25209GJ2017PLC097273",
    #         "invoice_number": "232431451",
    #         "invoice_date": "31/03/2024",
    #         "Party P.O. Ref": "AMAZON",
    #         "Date": "30/03/2024",
    #         "Date of Supply": "31/03/2024",
    #         "Transportation Mode": "",
    #         "Vehicle No": "",
    #         "Transporter Name": "",
    #         "LR No": "",
    #         "LR Date": ""
    #     }
    # },
    # "template_2": {
    #     "keys": {
    #         "invoice_number": ["Invoice No.","Invoice No", "Invoice Number", "Invoice #"],
    #         "invoice_date": ["Invoice Date","Invoice date"],
    #         "Buyer's Order No.": ["Buyer's Order No."],
    #         "Date": ["Date"],
    #         "Currency": ["Currency"],
    #         "Agent Code": ["Agent Code"],
    #         "Booking": ["Booking"],
    #         "Freight": ["Freight"],
    #         "LR No.": ["LR No."],
    #         "Transporter": ["Transporter"],
    #         "Final Destination": ["Final Destination"],
    #         "Vehicle No.": ["Vehicle No.","Vehicle Number"],
    #         "LR Date": ["LR Date"],
    #         "Payment Terms": ["Payment Terms"],
    #         "IRN No.": ["IRN No.","IRN Number","IRN NO."],
    #         "Acknowledgement No.": ["Acknowledgement No.","Acknowledge Number","Acknowledgement no."],
    #         "Acknowledgement Date": ["Acknowledgement Date","Acknowledge Date"],
    #     },
    #     "table_structure": {
    #         "columns": ["Sr No.", "Item Description", "HSN No.", "Batch No.", "No. of Pckgs", "Avg. Cont. in Kg.", "Total Qty in Kgs", "Unit Price/Kg.", "Taxable value"],
    #     },
    #     "expected_values": {
    #         "invoice_number": "ARDU125/853",
    #         "invoice_date": "12/09/2024",
    #         "Buyer's Order No.": "",
    #         "Date": "WR/PO/24-25/000616",
    #         "Currency": "12/09/2024",
    #         "Agent Code": "18",
    #         "Booking": "Godown Del",
    #         "Freight": "To Pay",
    #         "LR No.": "",
    #         "Transporter": "SURAT AHMEDABAD TRANS",
    #         "Final Destination": "BHIWANDI MAHARASHTRA",
    #         "Vehicle No.": "GJ16AW1936",
    #         "LR Date": "",
    #         "Payment Terms": "45 DAY PDC",
    #         "IRN No.": "22c31ac6a98f48063c5d7079de0c49ea1b450a343a2f68e1b122952a8c76c782",
    #         "Acknowledgement No.": "162418278014662",
    #         "Acknowledgement Date": "09/12/2024"
    #     }
    # },
    # "template_3": {
    #     "keys": {
    #         "Email": ["Email","E-Mail ID"],
    #         "Phone No": ["Phone No","Contact No"],
    #         "GSTIN": ["GSTIN","GST No","GSTIN Number","GST NO"],
    #         "CIN": ["CIN NO","CIN","CIN No"],
    #         "MSME": ["MSME"],
    #         "Reverse Charge": ["Reverse Charge"],
    #         "Invoice Number": ["Invoice Number","Invoice No.","Invoice NO"],
    #         "Invoice Date": ["Invoice Date"],
    #         "State": ["State"],
    #         "Transportation Mode": ["Transportation Mode"],
    #         "Vehicle Number": ["Vehicle Number","Vehicle No","Vehicle no."],
    #         "Date Of Supply": ["Date Of Supply"],
    #         "Place Of Supply": ["Place Of Supply"]
    #     },
    #     "table_structure": {
    #         "columns": ["Sr", "Description", "HSN/SAC", "GST Rate %", "Qty", "UOM", "Rate", "Dis. (%)", "Amount (INR)"],
    #     },
    #     "expected_values": {
    #         "Email": "screenotexprint@gmail.com",
    #         "Phone No": "02718666000, 02718666090",
    #         "GSTIN": "24AACCS6473L1Z0",
    #         "CIN": "U29120GJ1993PTCO19413",
    #         "MSME": "UDYAMGJ010007119",
    #         "Reverse Charge": "NO",
    #         "Invoice Number": "190",
    #         "Invoice Date": "01/09/2024",
    #         "State": "Gujarat",
    #         "Transportation Mode": "By Air",
    #         "Vehicle Number": "GJ01WE6723",
    #         "Date Of Supply": "01/09/2024",
    #         "Place Of Supply": ""
    #     }
    # },
    # "template_4":{
    #     "keys":{
    #         "Factory Add": ["Factory Add"],
    #         "Office Add":["Office Add"],
    #         "GST No":["GST No"],
    #         "Contact No":["Contact No"],
    #         "PAN No":["PAN No"],
    #         "E-mail ID":["E-mail ID"],
    #         "Vendor Name":["VENDOR NAME"],
    #         "Vendor Address":["VENDOR ADDRESS"],
    #         "Purchase order":["PURCHASE ORDER"],
    #         "Purchase PO Date":["PURCHASE PO DATE"],
    #         "PR No":["PR NO"],
    #         "Vendor Contact No":["VENDOR CONTACT NO"],
    #         "MSME":["MSME NO"]
    #     },
    #     "table_structure":{
    #         "columns":["SR. No.","Item Code","Item Description","QTY","Unit","Rate","Currency","Amount"]
    #     },
    #     "expected_values":{
    #         "Factory Add":"Plot No. B13/10/11/12/13 Hojiwala Ind. Estate Road No. 13",
    #         "Office Add": "Plot No. 179/B Road No. 6G Udhna Udyognagar Near Pastiwala.com Udhna Surat",
    #         "GST No":"24AAFFB8401R2Z7",
    #         "Contact No":"9825607334 ",
    #         "PAN No":"AAFFB8401R",
    #         "E-mail ID": "",
    #         "Vendor Name":"HERBERLEIN TECHNOLOGY AG",
    #         "Vendor Address": "9630 WATTTWIL",
    #         "Purchase order": "202425IM/242520002",
    #         "Purchase PO Date": "31/05/2024",
    #         "PR No": "",
    #         "Vendor Contact No":"41719874406",
    #         "MSME":""

    #     }
    # },
    # "template_5":{
    #         "keys":{
    #             "Document":["DOCUMENT"],
    #             "Challan No":["Challan No"],
    #             "Party Bill No":["Party Bill No"],
    #             "Document Date":["DOCUMENT DATE"],
    #             "Eway Bill No":["Eway Bill No"],
    #             "Removal Date Time":["Removal Date Time"],
    #             "GSTIN NO":["GSTIN NO."],
    #             "Veh.No.":["Veh.No."]
    #         },
    #         "table_structure":{
    #             "columns":["No.","Description","Hsn/Sac","Qty.","Unit","Rate","Disc. %","S.Gst %","C.Gst %", "Taxable Value"]
    #         },
    #         "expected_values":{
    #             "Document":"PR406",
    #             "Challan No":"8757",
    #             "Party Bill No.":"240022",
    #             "Document Date":"06/03/2024",
    #             "Eway Bill No.":"681693361290",
    #             "Removal Date Time":"06/03/2024 09:55AM",
    #             "GSTIN NO.":"24AAYFP7738E1Z0",
    #             "Veh.No.":"GJ05CU0904"
    #         }
    # },
    # "template_6":{
    #     "keys":{
    #         "Messers":["Messers"],
    #         "State":["State"],
    #         "GSTIIN/UIN":["GSTTIN/UIN"],
    #         "BILL NO.":["BILL NO"],
    #         "Issue Date":["DATE OF ISSUE"],
    #         "Job No.":["JOB NO"],
    #         "Job Date":["JOB Date"],
    #         "number_of_packs":["No. of Pack"],
    #         "Gr. Wt.":["Gr. Wt."],
    #         "Vessel":["VESSEL"],
    #         "Cif value":["CIF Value Rs"],
    #         "Port of Ship":["Port of Ship"],
    #         "BL No.":["BL No"],
    #         "Inv No.":["Inv No"],
    #         "B/E No":["B/E No"],
    #         "IGM Item No":["IGM ITEM No"],
    #         "Port of Del":["Port of Del"],
    #         "Description":["Description"],
    #         "Regd Office":["Regd Office"],
    #         "Pune Office":["Pune Office"],
    #         "Manglore":["Manglore"]
    #     },
    #     "table_structure":{
    #         "columns":["Particulars","SAC","CURR","QTY","RATE","AMOUNT","GST Rate","IGST","TOTAL AMOUNT"]
    #     },
    #     "expected_values":{
    #         "Messers":"BHAGAT TEXTILE ENGINEERS",
    #         "State":"Gujarat",
    #         "GSTIIN/UIN":"24AAFFB8401R2Z7",
    #         "BILL NO.":"SSL/NIM/085/23",
    #         "Issue Date":"3-Jun-23",
    #         "Job No.":"SSL/IMP/0032/23-24",
    #         "Job Date":"27-May-23",
    #         "number_of_packs":"1 PKG",
    #         "Gr. Wt.":"50.000 KGS",
    #         "Vessel":"",
    #         "Cif value":"1313442.00",
    #         "Port of Ship":"Zurich",
    #         "BL No.": "91016207925",
    #         "Inv No.":"90010205",
    #         "B/E No":"6144694",
    #         "IGM Item No":"2270072",
    #         "Port of Del":"Sahar Air Carg",
    #         "Description":"PARTS AND ACCESSORIES FOR EXTILE TEXTURISING MACHINE SWISSJE",
    #         "Regd Office":"4 Rustom Building 3rd Floor 29 Veer Nariman Road Near Akbarailys Fort Mumbai  400 001",
    #         "Pune Office": "Suyog Building",
    #         "Manglore":"Suja Nilaya Near Surya Narayana Temple Maroli Kulashekare Post Mangaiore"
    #     }
    # }
# }

# Convert PDF to images
# def convert_pdf_to_images(pdf_path, output_dir, dpi=300):
#     os.makedirs(output_dir, exist_ok=True)
#     try:
#         images = convert_from_path(pdf_path, dpi=dpi)
#     except Exception as e:
#         print(f"Error converting PDF to images: {e}")
#         return []

#     image_paths = []
#     for i, image in enumerate(images):
#         image_path = os.path.join(output_dir, f'page_{i + 1}.jpg')
#         image.save(image_path, 'JPEG')
#         image_paths.append(image_path)
#     return image_paths


def convert_pdf_to_images(pdf_path, output_dir, dpi=300):
    os.makedirs(output_dir, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    image_paths = []

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        # Get the page's pixmap (image)
        pix = page.get_pixmap(dpi=dpi)
        image_path = os.path.join(output_dir, f'page_{page_num + 1}.png')  # save as PNG
        pix.save(image_path)
        image_paths.append(image_path)

    return image_paths
# Extract text from images
def extract_text_from_images(image_paths, clean_text=True):
    full_text = ""
    for image_path in image_paths:
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            # print(f"Raw extracted text from {image_path}: {text}") 
            
            if clean_text:
                text = re.sub(r'[^\w\s\.-@]', '', text) 
            full_text += text + "\n"
        except Exception as e:
            print(f"Error extracting text from image {image_path}: {e}")
    return full_text

#                                                             DATA PREPROCESSING / DATA CLEANSING

# Prohibited date fields
prohibited_date_fields = [
    'Vendor Name', 'Vendor Address', 'Contact', 'Freight', 'Transporter', 
    'Booking', 'Agent Code', 'Acknowledgement No.', 'Party P.O. Ref'
]

# Forbidden keywords
forbidden_keywords = ["Acknowledgement Date", "Acknowledgement", "GSTIN", "Numbe"]

def remove_keys_from_value(value):
    all_keys = [re.escape(k) for keys in data_model.values() for k in keys]
    all_keys_pattern = r'\b(?:' + '|'.join(all_keys) + r')\b'
    
    # Remove any key or alias present in the value
    cleaned_value = re.sub(all_keys_pattern, '', value, flags=re.IGNORECASE).strip()
    return cleaned_value

def remove_forbidden_keywords(value):
    keyword_pattern = r'\b(?:' + '|'.join(map(re.escape, forbidden_keywords)) + r')\b'
    return re.sub(keyword_pattern, '', value, flags=re.IGNORECASE).strip()

allowed_date_fields = ["Acknowledgement Date", "Payment Terms"]

def extract_clean_date(value):
    date_match = re.search(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b', value)
    return date_match.group(1) if date_match else value  

def clean_numeric_value(value):
    # Remove commas from the value and convert to float
    return str(float(value.replace(',', ''))) if value else ""

def clean_value(value, field_name):
    # Call clean_numeric_value for specific fields
    if field_name in ['rate', 'amount', 'discount', 'taxable_value', 'value']:
        return clean_numeric_value(value)
    return value.strip()

def find_value_in_text(text, key_aliases, field_name, pdf_path):
    all_keys = [re.escape(k) for keys in data_model.values() for k in keys]
    all_keys_pattern = r'\b(?:' + '|'.join(all_keys) + r')\b'

    for alias in key_aliases:
        pattern = re.escape(alias) + r'\s*[:\-]?\s*(.*?)(?=\s*\|)?(?=\n(?!.*' + all_keys_pattern + r')|(?=\n|$))'
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            value = match.group(1).strip()
            start = match.start(1)
            end = match.end(1)
            value = value.rstrip('.')

            cleaned_value = clean_value(value, field_name)
            cleaned_value = remove_forbidden_keywords(cleaned_value)

            # Check if the cleaned value contains any other key or alias
            for key in all_keys:
                if re.search(r'\b' + key + r'\b', cleaned_value, re.IGNORECASE):
                    cleaned_value = re.split(r'\b' + key + r'\b', cleaned_value, maxsplit=1, flags=re.IGNORECASE)[0].strip()

            coordinates = get_coordinates_from_pdf(pdf_path, alias)

            return {"value": cleaned_value, "coordinates": coordinates}

    # Fallback pattern
    for alias in key_aliases:
        fallback_pattern = re.escape(alias) + r'\s*[:\-]?\s*(.*?)\s*$'
        fallback_match = re.search(fallback_pattern, text, re.IGNORECASE)
        if fallback_match:
            value = fallback_match.group(1).strip()
            cleaned_value = clean_value(value, field_name)
            cleaned_value = remove_forbidden_keywords(cleaned_value)

            # Check again for any key or alias in the value
            for key in all_keys:
                if re.search(r'\b' + key + r'\b', cleaned_value, re.IGNORECASE):
                    cleaned_value = re.split(r'\b' + key + r'\b', cleaned_value, maxsplit=1, flags=re.IGNORECASE)[0].strip()

            coordinates = get_coordinates_from_pdf(pdf_path, alias)

            return {"value": cleaned_value, "coordinates": coordinates}

    return {"value": "", "coordinates": None}

def get_coordinates_from_pdf(pdf_path, alias):
    pdf_document = fitz.open(pdf_path)
    coordinates = {"page": None, "x": None, "y": None, "width": None, "height": None}

    #finding the alias directly first
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text_instances = page.search_for(alias)

        if not text_instances:
            # If alias not found, check if it's a number with commas or without commas
            if re.match(r'[\d,]+(?:\.\d{2})?', alias):
                # If alias is a number, try both with commas and without commas
                alias_without_commas = alias.replace(',', '')
                text_instances = page.search_for(alias_without_commas)

            if not text_instances:
                print(f"Alias '{alias}' not found on page {page_num + 1}")
                continue

        for inst in text_instances:
            rect = fitz.Rect(inst)
            coordinates = {
                "page": page_num + 1,
                "x": rect.x0,
                "y": rect.y0,
                "width": max(1, rect.width),
                "height": max(1, rect.height)
            }
            break 

        if coordinates["x"] is not None:
            break

    pdf_document.close()
    return coordinates if coordinates["x"] is not None else None

def get_coordinates_for_numeric_fields(pdf_path, field_name, numeric_value):
    pdf_document = fitz.open(pdf_path)
    coordinates = {"page": None, "x": None, "y": None, "width": None, "height": None}

    # List of possible labels for numeric fields
    possible_labels = [field_name.lower(), field_name.upper(), field_name.capitalize()]

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]

        # First, search for possible labels (e.g., 'Rate' or 'Amount')
        for label in possible_labels:
            label_instances = page.search_for(label)
            if label_instances:
                for inst in label_instances:
                    # Look for the numeric value in close proximity to the label
                    label_rect = fitz.Rect(inst)
                    numeric_instances = page.search_for(numeric_value)

                    if not numeric_instances:
                        # If numeric value not found, check without commas
                        numeric_instances = page.search_for(numeric_value.replace(",", ""))

                    # Check if any numeric value is close to the label
                    for num_inst in numeric_instances:
                        num_rect = fitz.Rect(num_inst)
                        if label_rect.y0 <= num_rect.y0 <= label_rect.y1 + 100:  # Nearby vertically
                            coordinates = {
                                "page": page_num + 1,
                                "x": num_rect.x0,
                                "y": num_rect.y0,
                                "width": num_rect.width,
                                "height": num_rect.height
                            }
                            break
                    if coordinates["x"] is not None:
                        break

        if coordinates["x"] is not None:
            break

    pdf_document.close()
    return coordinates if coordinates["x"] is not None else None

def get_row_coordinates_from_pdf(pdf_path, row_data):
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

    for field, field_data in row_data.items():
        if isinstance(field_data, dict) and field_data.get("coordinates"):
            coords = field_data["coordinates"]
            if coords and coords["x"] is not None and coords["y"] is not None:
                min_x = min(min_x, coords["x"])
                min_y = min(min_y, coords["y"])
                max_x = max(max_x, coords["x"] + coords["width"])
                max_y = max(max_y, coords["y"] + coords["height"])

    if min_x < float('inf') and min_y < float('inf'):
        return {
            "page": row_data.get("sr_no", {}).get("coordinates", {}).get("page", 1),
            "x": min_x,
            "y": min_y,
            "width": max_x - min_x,
            "height": max_y - min_y
        }
    return None

def get_combined_row_coordinates(all_row_data):
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

    for row_data in all_row_data:
        row_coords = row_data.get("row_coordinates", {})
        if row_coords and row_coords["x"] is not None and row_coords["y"] is not None:
            min_x = min(min_x, row_coords["x"])
            min_y = min(min_y, row_coords["y"])
            max_x = max(max_x, row_coords["x"] + row_coords["width"])
            max_y = max(max_y, row_coords["y"] + row_coords["height"])

    if min_x < float('inf') and min_y < float('inf'):
        return {
            "page": all_row_data[0].get("sr_no", {}).get("coordinates", {}).get("page", 1),
            "x": min_x,
            "y": min_y,
            "width": max_x - min_x,
            "height": max_y - min_y
        }
    return None

patterns = [
    r'^\s*(?P<sr_no>\d+)\s+(?P<product_name>.+?)\s+(?P<product_code>\w+)\s+(?P<hsn_code>\d+)\s+' \
    r'(?P<qty>\d+(\.\d+)?)\s+(?P<uom>[A-Z]+)\s+(?P<rate>[\d,.]+)\s+(?P<amount>[\d,.]+)\s+' \
    r'(?P<discount>[\d,.]*)\s+(?P<taxable_value>[\d,.]+)\s+(?P<value>[\d,.]+)?\s*$',

    r'^\s*(?P<sr_no>\d+)\s+(?P<item_description>.+?)\s+(?P<hsn_no>\d{4}\.\d{2}\.\d{2})\s+(?P<batch_no>\d+)\s+' \
    r'(?P<no_of_pckgs>[\d.]+)\s+(?P<avg_cont_in_kg>[\d.]+)\s+(?P<total_qty_in_kgs>[\d,.]+)\s+' \
    r'(?P<unit_price_kg>[\d,.]+)\s+(?P<taxable_value>[\d,.]+)\s*$',

    r'^\s*(?P<sr>\d+)\s+(?P<description>.+?)\s+(?P<hsn_sac>\d{4}\.\d{2}\.\d{2})?\s+(?P<gst_rate>[\d.]+)?\s+' \
    r'(?P<qty>[\d,.]+)\s+(?P<uom>[A-Z]+)\s+(?P<rate>[\d,.]+)\s+(?P<discount>[\d,.]+)?\s*' \
    r'(?P<amount_inr>[\d,.]+)\s*$',

    r'^\s*(?P<sr_no>\d+)\s+(?P<item_code>.+?)\s+(?P<item_description>.+?)\s+(?P<qty>[\d.]+)\s+' \
    r'(?P<unit>[A-Za-z]+)\s+(?P<rate>[\d,.]+)\s+(?P<currency>[\w]+)\s+(?P<amount>[\d,.]+)\s*$',

    r'^\s*(?P<no>\d+)?\s+(?P<description>.+?)?\s+(?P<hsn_sac>\w+)?\s+(?P<qty>[\d.]+)?\s+' \
    r'(?P<unit>[A-Za-z]+)?\s+(?P<rate>[\d,.]+)?\s+(?P<discount>[\d,.]+)?\s+' \
    r'(?P<sgst>[\d,.]+)?\s+(?P<csgt>[\d,.]+)?\s+(?P<taxable_value>[\d,.]+)?\s*$',

    r'^\s*(?P<particulars>.+?)\s+(?P<sac>\w+)\s+(?P<curr>\w+)\s+(?P<qty>[\d,.]+)\s+' \
    r'(?P<rate>[\d,.]+)\s+(?P<amount>[\d,.]+)\s+(?P<gst_rate>[\d,.]+)\s+(?P<igst>[\d,.]+)\s+(?P<total_amount>[\d,.]+)\s*$'
]

# def extract_table_data(text, pdf_path,temp):
#     table_data = []
#     rows = text.splitlines()
#     unique_rows = set()

#     for row in rows:
#         row = row.strip()
#         row_data = None

#         # Try matching row with each pattern until one matches
#         for pattern in patterns:
#             match = re.match(pattern, row)
#             if match:
#                 row_data = match.groupdict()
#                 break

#         # If a pattern matches, proceed to process the row data
#         if row_data:
#             sr_no = row_data.get("sr_no") or row_data.get("sr")
#             if sr_no and sr_no in unique_rows:
#                 continue  # Skip duplicate rows based on serial number

#             unique_rows.add(sr_no)
#             row_data = {k: (v.strip() if v else "") for k, v in row_data.items()}

#             row_data_with_coords = {}
#             for field, value in row_data.items():
#                 alias = value.strip()
#                 coordinates = get_coordinates_from_pdf(pdf_path, alias)
#                 row_data_with_coords[field] = {
#                     "value": value,
#                     "coordinates": coordinates or {"page": None, "x": None, "y": None, "width": None, "height": None}
#                 }

#             row_data_with_coords["row_coordinates"] = get_row_coordinates_from_pdf(pdf_path, row_data_with_coords)
#             table_data.append(row_data_with_coords)

#     combined_coordinates = get_combined_row_coordinates(table_data)
#     return table_data, combined_coordinates

def extract_table_data(text, pdf_path, template):
    table_structure = template.get('table_structure', {}).get('columns', [])
    # print("table_structure=========> ",table_structure)
    if not table_structure:
        raise ValueError("Template does not contain 'table_structure'.")

    # The rest of your logic
    table_data = []
    rows = text.splitlines()
    unique_rows = set()

    for row in rows:
        row = row.strip()
        row_data = None

        for pattern in patterns:  # Using patterns from the template
            match = re.match(pattern, row)
            if match:
                row_data = match.groupdict()
                break

        if row_data:
            sr_no = row_data.get("sr_no") or row_data.get("sr")
            if sr_no and sr_no in unique_rows:
                continue

            unique_rows.add(sr_no)
            row_data = {k: (v.strip() if v else "") for k, v in row_data.items()}

            row_data_with_coords = {}
            for field, value in row_data.items():
                alias = value.strip()
                coordinates = get_coordinates_from_pdf(pdf_path, alias)
                row_data_with_coords[field] = {
                    "value": value,
                    "coordinates": coordinates or {"page": None, "x": None, "y": None, "width": None, "height": None}
                }

            row_data_with_coords["row_coordinates"] = get_row_coordinates_from_pdf(pdf_path, row_data_with_coords)
            table_data.append(row_data_with_coords)
            # print("----------> ",table_data)
    combined_coordinates = get_combined_row_coordinates(table_data)
    return table_data, combined_coordinates


def extract_invoice_data_with_table(pdf_path, text, template):
    # print("Template : ",template)
    if 'keys' not in template or 'table_structure' not in template:
        raise ValueError("Template must contain 'keys' and 'table_structure'.")

    extracted_data = {}

    for key, aliases in template['keys'].items():
        result = find_value_in_text(text, aliases, key, pdf_path)
        value = result['value']
        
        # If the value is empty, set the coordinates to None
        if not value:
            extracted_data[key] = ""
            extracted_data[f"{key}_coordinates"] = {"page": None, "x": None, "y": None, "width": None, "height": None}
        else:
            extracted_data[key] = value
            extracted_data[f"{key}_coordinates"] = result['coordinates']

    table_data, combined_row_coordinates = extract_table_data(text, pdf_path,template)
    print("\n+++++++",table_data, combined_row_coordinates)
    extracted_data['Table Data'] = table_data
    extracted_data['Combined Row Coordinates'] = combined_row_coordinates
    return extracted_data

def calculate_accuracy(extracted_data, template):
    if 'expected_values' not in template:
        raise ValueError("Template must contain 'expected_values'.")
    matched_fields = sum(1 for key, value in template['expected_values'].items() if extracted_data.get(key) == value)
    total_fields = len(template['expected_values'])
    return (matched_fields / total_fields) * 100 if total_fields else 0

def process_invoice_with_all_templates(pdf_path, output_dir,templates):
    image_paths = convert_pdf_to_images(pdf_path, output_dir)
    extracted_text = extract_text_from_images(image_paths)

    all_results = {}
    # for template_name, template in templates.items():
    #     extracted_data = extract_invoice_data_with_table(pdf_path, extracted_text, template)
    #     accuracy = calculate_accuracy(extracted_data , template)
    #     all_results[template_name] = {
    #         "extracted_data": extracted_data,
    #         "accuracy": accuracy
    #     }
    for template_name, template in templates.items():
        try:
            # print(f"Processing {template_name} with data: {template}")
            extracted_data = extract_invoice_data_with_table(pdf_path, extracted_text, template)
            # print("\n------extracted_data : ",extracted_data)
            accuracy = calculate_accuracy(extracted_data, template)
            all_results[template_name] = {
                "extracted_data": extracted_data,
                "accuracy": accuracy
            }
        except Exception as e:
            print(f"Error processing template {template_name}: {e}")
            all_results[template_name] = {
                "error": str(e),
                "accuracy": 0
            }

    return all_results

# if __name__ == "__main__":
#     pdf_path = r'/Users/bhavyadalal/Documents/reactqit/invoice/invoice_7.pdf'
#     # print(pdf_path)
#     output_dir = r'./'
#     # pdf_path = r'C:\Projects\invoice_annotation\invoices\invoice_8.pdf'
#     # output_dir = r'C:\Projects\invoice_annotation\invoices\images'
    
#     all_template_results = process_invoice_with_all_templates(pdf_path, output_dir)

#     output_json_path = os.path.join(output_dir, 'all_template_results.json')
#     with open(output_json_path, 'w') as json_file:
#         json.dump(all_template_results, json_file , indent=6)

#     # print(f"All template extraction results saved to {output_json_path}")

# # Saving the results to a .pkl file
# with open('invoice_data.pkl', 'wb') as file:
#     pickle.dump(all_template_results, file)

# # print("Results have been saved to 'invoice_data.pkl'")

# # Load the .pkl file
# with open('invoice_data.pkl', 'rb') as file:
#     invoice_data = pickle.load(file)

# # Use the loaded data (you can access any field now)
# # print(invoice_data)

