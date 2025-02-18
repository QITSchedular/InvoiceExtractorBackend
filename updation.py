
import re
import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import json
import numpy as np
import cv2
import fitz
import requests  
import camelot
import os
from fuzzywuzzy import fuzz
os.environ["CONDA_PREFIX"] = "E:/conda_envs/camelot_env"

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
        img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)  # Change blockSize to 11

    # Optional: Erosion and dilation to enhance text regions
    kernel = np.ones((3, 3), np.uint8)
    img_dilated = cv2.dilate(img_thresh, kernel, iterations=1)
    img_eroded = cv2.erode(img_dilated, kernel, iterations=1)

    # Convert back to PIL image
    preprocessed_img = Image.fromarray(img_eroded)
    return preprocessed_img

# Data Model                  # replace it 


data_model =  {
    "keys": {
      "invoice_number": [
        "Invoice No",
        "Invoice Number",
        "Invoice #",
        "Inv No.",
        "invoice_number",
        "Invoice No.",
        "invoice No." ,
        "nvoice No.",
        "nvoice No"
      ],
      "Company Name":["Company Name","Company"],
      "invoice_date": ["Invoice Date", "Dated","invoice_date","Invoice Date "],
      "vendor_name": [
        "Vendor Name",
        "VENDOR NAME",
        "Customer Name",
        "Buyer Name",
        "vendor_name"
      ],
      "P 0. No. & Date":["P 0. No. & Date"],
      "GST NO": [
        "GST No",
        "GST NO",
        "GSTIN Number",
        "GSTIN Numbe",
        "GSTTIN/UIN",
        "GSTIN No",
        "GSTIN",
        "GSTIN:"
      ],
      "PAN No": ["PAN No", "PAN Number","PAN NO","PAN NO:","PAN"],
      "State Code": [
        "State Code",
        "StateCode",
        "statecode",
        "state-code",
        "stateCode",
        "Statecode",
        "state code"
      ],
      "Vendor Contact": ["VENDOR CONTACT NO", "Buyer Contact","Vendor Contact"],
      "E-Mail": ["Email", "E-mail ID", "email","E-Mail","E-mail Id", "Email:","email:","Email ID","E-mail Id"],
      "Total Amount": [
        "Total Amount",
        "Grand Total",
        "Total",
        "Total Payble Amount",
        "Net Amount",
        "Basic Amount",
        "Tax Amount",
        "Grand Total INR",
        "Total Amount: INR",
        "Grand Total: INR",
        "Tax Amount(in words) INR"
      ],
      "Amount in words":["Amount In Words"],
      "Packaging":["PACKAGING","Packing"],
      "After Tax Amount": ["After Tax", "Amount After Tax","Rupees"],
      "place_of_supply": ["Place of Supply", "Supply Place", "Place Of Suppy","place_of_supply"],
      "CIN": ["CIN No", "CIN Number", "cin number", "Cin number", "CIN NO","CIN","CINNo","CINNo."],
      "Currency": ["Currency", "currency"],
      "Vehicle No.": ["Vehicle No", "Vehicle No.", "Vehicle Number", "Veh.No.","Motor Vehicle No."],
      "Agent Code": ["Agent Code", "agentcode", "agent code"],
      "IRN No.": ["IRN No.", "IRN No", "IRN Number", "IRN NO.","IRN"],        
      "Payment Terms": ["Payment Terms","Mode/Terms of Payment","Payment Terms:","Payment Terms :"],        
      "Acknowledgement No.": [
        "Acknowledgement No",
        "Acknowledge Number",
        "Ack. No",
        "Ack. Number",
        "Acknowledgement No.",
        "Ack No."           
      ],
      "Acknowledgement Date": ["Acknowledge Date", "Ack. Date","Ack Date"],      
      "Transporter": ["Transporter", "Transporter Name"], 
      "Transportation Mode": ["Transportation Mode","Mode of Transport"],
      "Factory address": ["Factory Address", "Factory Add"],
      "Address":["Adress","Address","address"],
      "Office address": [
        "Office Address",
        "Office Add",
        "Office",
        "Office address"
      ],
      "Vendor Address": ["Vendor Address"],
      "Billing Address":["Billing Address","Blling Address","Biling Address"],
      "Contact Person":["Contact person","Contact Peraon"],
      "LR No": ["LR No", "LR no", "LR No.", "LR Number"],
      "LR Date": ["LR Date", "LR date"],
      "Party P.O. Ref": ["Party P.O. Ref", "Party P.O. ref"],
      "Purchase Order": ["PURCHASE ORDER", "Purchase order","PURCHASE ORDER No","Purchase Order No"],
      "Order No":["Order No","Order No."],
      "Order Date":["Order Date"],
      "PO No.":["PO No.","PO No","PO Number","PO No :","PO No:","PO.No.","P.O. No."],      
      "PO Date": ["PO Date","PO Date:","P.O. Date"],    
      "Purchase PO Date": ["PURCHASE PO DATE", "Purchase PO Date"],
      "Declaration": ["Declaration", "declaration"],
      "Destination": ["Destination", "Final Destination", "Final"],            
      "Booking": ["Booking", "booking"],
      "Bill Date": ["eWay Bill Date", "eWay Bill Valid Date","Bill Date","Bill Date >"],
      "Challan Date": ["Challan Date"],
      "Machine Number": [
        "Machine No.",
        "Machine Number",
        "Machine no.",
        "Machine No"
      ],
      "Date Of Supply": ["Date Of Supply", "date_of_supply", "Supply date"],
      "Date": ["Date", "date","Dated"],       
      "State": ["State", "state", "STATE","State Name"],
      "Party Bill No": ["Party Bill No"],
      "Eway Bill No.": ["Eway Bill No.","e-Way Bill No.","E-way Bill No","EWay Bil No","EWay Bil No.","e-Way Bil No.","e-Way Bil No"],  
      "Removal Date Time": ["Removal Date Time"],
      "Challan No": ["Challan No","Challan No."],
      "Document": ["DOCUMENT", "Document", "document"],
      "Dispatch Doc No.":["Dispatch Doc No."],    
      "Document No":["Document No","Doc No."],      
      "Insurance":["Insurance"],
      "Delivery":["Delivery"],
      "Delivery terms":["Delivery terms"],
      "Delivery Note ":["Delivery Note"],
      "Delivery Note Date":["Delivery Note Date"],
      "Document Date": ["DOCUMENT Date", "Document Date"],
      "Messers": ["Messers"],
      "BILL NO.": ["BILL NO","Bill No"],
      "Issue Date": ["DATE OF ISSUE", "Issue Date"],
      "Job No.": ["JOB NO", "Job No."],
      "Location Code":["Location Code"],
      "Job Date": ["JOB Date", "Job date","Job Date"],
      "number_of_packs": ["No. of Pack", "no._of_packs","number_of_packs"],
      "Gr. Wt.": ["Gr. Wt."],
      "Vessel": ["VESSEL", "vessel","Vessel"],
      "Cif value": ["CIF Value Rs", "Cif value"],
      "Port of Ship": ["Port of Ship"],
      "BL No.": ["BL No", "BL No."],
      "B/E No": ["B/E No"],
      "IGM Item No": ["IGM ITEM No", "IGM_Item_no", "IGM_ITEM_NO","IGM Item No"],
      "Port of Del": ["Port of Del",],
      "Description": ["Description", "description"],
      "Manglore": ["Manglore"],
      "Receiver Details": [
        "Receiver Details Billed to",
        "Receiver Details",
        "Billed to",
        "Consignee Details",
        "Buyer Name",
        "Consignee Details",
        "Buyer's Details",
        "Bill To",
        "Buyer (Bill to)",
        "Bill  To",
        "Blling To",
        "Biling",
        "Blling"
      ],
      "Duplicate for Supplier": ["Duplicate for Supplier", "Extra Copy"],
      "Vendor contact no": ["VENDOR CONTACT NO", "VENDOR", "Vendor Contact No","Vendor contact no"],
      "Sr no": ["SR.", "SR No.", "No.","Sr no","Sr.","Sr No"],
      "Total Amount Before Tax": ["Total Amount Before Tax", "Basic Amount"],
      "Add : Transport": ["Add : Transport"],
      "Taxable Amount": ["Taxable Amount", "taxable amount"],
      "Round Off": ["Round Off", "Round off"],
      "Warranty":["Warranty","WARRANTY"],
      "Total Amount After Tax": [
        "After Tax",
        "Amount After Tax",
        "Total Amount After Tax",
        "Grand Total",
        "Amount Chargeable", 
        "Amount Chargeable(in words)"          

      ],
      "Qtn No":["Qtn. No."],
      "Qtn date":["Qtn. Date"],
      "Kind Attn.":["Kind Attn."],
      "M/S.":["M/S."],
      "Bank":["Bank"],
      "Account No":["Account No"],
      "Branch":["Branch"],
      "Insurance Terms":["Insurance Terms"],
      "Freight": ["FREIGHT", "Freight", "FREIGHT IMP/EXP-AIR","Freight Terms"],
      "Regd. Office": ["Regd. Office","REGD.ADD"],
      "Factory": ["Factory"],
      "Phone": ["Contact No", "Phone", "Phone No", "Contact","Mobile","Mob.","Mob","Phone :","Phone:","Cell No.","Cell No","Mobile No.","Mobile No"],
      "Reference":["Relerence","Reference"],    
      "Website": ["Website", "website"],
      "Payment Mode": ["Payment Mode", "mode_of_payment","Mode/Terms of Payment"],
      "Order Ref. No.": ["Order Ref. No."],
      "Buyer's Order No.": ["Buyer's Order No.","Buyers Order No."],
      "Sub Total": ["Sub Total","Sub","SuD Total"],
      "MSME": ["MSME", "Msme", "msme"],
      "Reverse Charge": ["reverse charge", "Reverse Charge"],
      "PR No": ["PR No", "pr no", "Pr No", "PRNO"],
      "Buyer GST NO": ["Buyer GSTIN", "Buyer GST No"],
      "Name of Product/Service": [
        "Name of Product/Service",
        "Product/Service Description",
        "Item Name",
        "Item Description",
        "Service/Product Title",
        "Description of Goods/Services",
        "Goods/Service Name",
        "Item Details",
        "Product/Service Info",
        "Line Item Description",
        "Details of Product/Service"
      ],
      "Product Code": [
        "Product Code",
        "Item Code",
        "Product ID",
        "Item ID",
        "Catalog Number",
        "Item Number",
        "Stock Code",
        "Inventory Code",
        "Model Number",
        "Reference Code",
        "Product Reference"
      ],
      "HSN/SAC Code": [
        "HSN/SAC Code",
        "HSN No.",
        "HSN Code",
        "SAC Code",
        "Harmonized System of Nomenclature (HSN)",
        "Service Accounting Code (SAC)",
        "Tax Code",
        "Product Classification Code",
        "Service Classification Code",
        "HS Code",
        "Customs Code",
        "Item Tax Code",
        "Classification Code",
        "Goods/Service Code",
        "HSN/SAC Classification",
        "HSN/SAC Number",
        "HSN/SAC"
      ],
      "QTY.": [
        "QTY",
        "Quantity",
        "Item Quantity",
        "Product Quantity",
        "No. of Items",
        "Total Quantity",
        "Ordered Quantity",
        "Quantity Ordered",
        "Quantity of Product",
        "Item Count",
        "Units Ordered",
        "Quantity of Items",
        "Product Units",
        "QTY."
      ],
      "UOM": [
        "UOM",
        "Unit of Measure",
        "Measurement Unit",
        "Unit",
        "Unit Type",
        "Measurement Type",
        "Unit Quantity",
        "Product Unit",
        "Service Unit",
        "Quantity Unit",
        "Measurement Unit Code",
        "Product Measurement Unit",
        "Service Measurement Unit",
        "UOM Code"
      ],
      "Rate": [
        "Rate",
        "Unit Price",
        "Price per Unit",
        "Item Rate",
        "Product Rate",
        "Cost per Unit",
        "Basic Rate"
      ],
      "Amount": [
        "Amount",
        "Total Amount",
        "Invoice Amount",
        "Net Amount",
        "Total Price",
        "Amount Due"
      ],
      "Discount": [
        "Discount",
        "Price Reduction",
        "Discounted Price",
        "Total Discount",
        "Discount Amount",
        "Rebate",       
        "Disc%"
      ],
      "Taxable Value": [
        "Taxable Value",
        "Taxable Amount",
        "Taxable Price",
        "Taxable Total",
        "Value for Tax",
        "Net Taxable Value"
      ],
      "Value": [
        "Value",
        "Total Value",
        "Item Value",
        "Net Value",
        "Gross Value",
        "Invoice Value",
        "Total Payable Amount"
      ],
      "Item Description": [
        "Item Description",
        "Product Description",
        "Service Description",
        "Description",
        "Item Details",
        "Product Details",
        "Particulars",
        "Description of Goods",
        "Item Desc"
      ],
      "Batch No.": [
        "Batch No.",
        "Batch Number",
        "Lot No.",
        "Lot Number",
        "Manufacturing Batch",
        "Production Batch"
      ],
      "No. of Pckgs": [
        "Number of Packages.",
        "Package Count",
        "No. of Boxes",
        "Package Units"
      ],
      "Avg. Cont. in Kg.": [
        "Average Weight",
        "Avg Weight",
        "Average Content",
        "Avg. Weight per Package",
        "Average Kg per Unit",
        "Avg. Cont. in Kg."
      ],
      "Total Qty in Kgs": [
        "Total Weight",
        "Total Quantity",
        "Total Kg",
        "Total Weight in Kilograms",
        "Total Weight (in Kgs)",
        "Total Qty in Kgs"
      ],
      "Unit Price/Kg.": [
        "Price per Kg",
        "Unit Price (Kg)",
        "Cost per Kg",
        "Price per Kilogram",
        "Kg Unit Price",
        "Unit Price/Kg."
      ],
      "Taxable value": [
        "Taxable Amount",
        "Value for Tax",
        "Taxable Base",
        "Taxable Price",
        "Amount Subject to Tax",
        "Taxable value"
      ],
      "GST Rate %": [
        "GST Percentage",
        "GST Rate",
        "GST (%)",
        "GST % on Product",
        "GST Rate %",
        "GST"
      ],
      "S.Gst %": [
        "State GST Rate",
        "SGST Percentage",
        "SGST (%)",
        "State GST %",
        "S-GST Rate",
        "SGST %",
        "State GST Percentage",
        "S.Gst %",
        "SGST(9.00%)"
      ],
      "Value of State GST Rs.": [
        "SGST Amount",
        "State GST Amount",
        "SGST Value",
        "Value of SGST",
        "Value of State GST Rs."
      ],
      "CGST %": [
        "Central GST Rate",
        "CGST Percentage",
        "CGST (%)",
        "Central GST %",
        "C-GST Rate",
        "CGST %",
        "Central GST Percentage",
        "C.Gst %",
        "CGST"
      ],
      "Rate with Tax": [
        "Tax Inclusive Price",
        "Price with Tax",
        "Total Price (Including Tax)",
        "Price After Tax",
        "Amount with Tax",
        "Rate with Tax"
      ],
      "IGST": [
        "IGST",
        "Integrated GST",
        "GST on Interstate Supply",
        "Interstate GST",
        "IGST Tax",
        "IGST Amount",
        "GST on Import"
      ],
      "Reference No. & Date.":[                 
          "Reference No. & Date."
      ],
      "Other References":[                      
          "Other References"
      ],
      "Delivery Through": [
                "Delivery Through","Dispatched through"           
            ],
    "Bill of Lading/LR-RR No.":[
        "Bill of Lading/LR-RR No."                 
    ],
    "Test Report":["Test Report"]
    },      # Regex            
    "regex": {
      "invoice_date": "\\d{2}[-/]\\d{2}[-/]\\d{4}",
      "vendor_name": "^[A-Za-z\\s\\.,&']+$",
      "GST NO": "\d{2}[A-Z]{5}\d{4}[A-Z]{1}[0-9A-Z]{1}Z[0-9A-Z]",
      "PAN No": "[A-Z]{5}[0-9]{4}[A-Z]{1}$|^\.\:\s[A-Z]{5}\d{4}",
      "State Code": "\\d{2}",
      "Vendor Contact": "\\d{10}",
      "E-Mail": "[a-zA-Z0-9._%+-]+\s*@\s*[a-zA-Z0-9.-]+\s*\.\s*[a-zA-Z]{2,3}", 
      "Total Amount": "\\d+(\\.\\d{1,2})?",
      "After Tax Amount": "\\d+(\\.\\d{1,2})?",
      "place_of_supply": "[A-Za-z\\s]+",
      "CIN": "[A-Z]{1}[0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3,4}[0-9]{5}",
      "Currency": "[A-Za-z]{3}",
      "Vehicle No.": "([A-Z]{2}-\d{1,2}[A-Z]{0,2}-\d{1,4}|\d{2}[A-Z]{1,2}\d{1,4}[A-Z0-9]{0,3})",
      "Agent Code": "[A-Za-z0-9\\-]+",
      "IRN No.": "[0-9a-fA-F]{64}|[0-9a-fA-F]{32,64}|[0-9a-fA-F]{32,64}",
      "LR No": "[A-Za-z0-9\\-/]+",
      "LR Date": "\\d{2}/\\d{2}/\\d{4}",
      "Purchase Order": "[A-Za-z0-9\\-/]+",
      "Purchase PO Date": "\\d{2}[-/]\\d{2}[-/]\\d{4}",
      "Bill Date": "\\d{2}/\\d{2}/\\d{4}",
      "Date Of Supply": "\\d{2}[-/]\\d{2}[-/]\\d{4}",
      "Date": "\\d{2}[-/]\\d{2}[-/]\\d{4}",
      "State": "[A-Za-z0-9\s\-\+]+",
      "Removal Date Time": "\\d{2}/\\d{2}/\\d{4} \\d{2}:\\d{2}",
      "Challan No": "[A-Za-z0-9\\-]+",
      "Document": "[A-Za-z0-9\\s]+",
      "Document Date": "\\d{2}/\\d{2}/\\d{4}",
      "Messers": "[A-Za-z0-9\\s]+",
      "BILL NO.": "[A-Za-z0-9\\-]+",
      "Issue Date": "\\d{2}/\\d{2}/\\d{4}",
      "Sr no": "\\d+",
      "Phone": "(\d{10}|(\d{5}-\d{6})(?:,\s*\d{5}-\d{6})*)[,]?$|^\d{4}-\d{7}-\d{7}-\d{10}|\d{4}-\d{7}\.\s*\d{7}",
      "Website": "(https?://)?(www\\.)?[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}",
      "Order Ref. No.": "[A-Za-z0-9\\-]+",
      "Buyer's Order No.": "\s*[A-Za-z]+\/[A-Za-z]+\/\d{2}-\d{2}\/\d{6}", 
      "MSME": "(?:\d{2}[A-Z]{5}\d{4}[A-Z]{1}[0-9A-Z]{1}[Z]{1}[0-9A-Z]{1}|UDYAM-[A-Z]{2}-\d{2}-\d{7})",
      "PR No": "[A-Za-z0-9\\-/]+",
      "Buyer GST NO": "\\d{2}[A-Z]{5}\\d{4}[A-Z]{1}[0-9A-Z]{1}[Z]{1}[0-9A-Z]{1}",
      "Acknowledgement Date":"\d{2}[-/]\d{2}[-/]\d{4}"
    }
  }

# Templates             

templates = {                          
    "template_1": {                    
        "keys": {
            "Date":  [
                "Date",
                "date",
                "Dated"
            ],  
            "LR No" :  [
                "LR No.",
                "LR no",
                "LR No",
                "LR Number"
            ], 
            "CIN": [
                "CIN No",
                "CIN Number",
                "cin number",
                "Cin number",
                "CIN NO"
                ],

            "GST NO": [
                "GSTTIN/UIN",
                "GSTIN No",
                "GSTIN Number",
                "GST No",
                "GST NO",
                "GSTIN",
                "GSTIN Numbe"
            ],
            "PAN No": [
                "PAN No",
                "PAN Number",
                "PAN NO."
            ],
            "LR Date": [
                "LR date",
                "LR Date"
            ],
            "Round Off": [
                "Round Off",
                "Round off" 
            ],
            "Vehicle No.": [
                "Vehicle No",
                "Vehicle Number",
                "Vehicle NO.",
                "Vehicle No.",
                "Veh.No.",
                "Motor Vehicle No."
            ],
            "invoice_date": [
                "Invoice date",
                "Invoice Date",
                "Dated"
            ],
            "Date Of Supply": [
                "Supply Date",
                "Date of Supply",
                "Date Of Supply",
                "Date of supply",
                "Supply date",
                "date_of_supply"
            ],
            "Party P.O. Ref": [
                "Party P.O. Ref",
                "Party P.O. ref"
            ],
            "Taxable Amount": [
                "Taxable Amount",
                "taxable amount"
            ],
            "invoice_number": [
                "Invoice #",
                "Invoice No",
                "Invoice Number",
                "Inv No.",
                "Invoice No."
            ],
            "Add : Transport": [
                "Add : Transport"
            ],
            "Transporter": [
                "Transporter Name",
                "Transporter"
            ],
            "Transportation Mode": [
                "Transportation Mode"
            ],
            "Total Amount After Tax": [
                "After Tax",
                "Amount After Tax",
                "Total Amount After Tax",
                "Grand Total",
                "Amount Chargeable", 
                "Amount Chargeable(in words)"

            ],
            "Total Amount Before Tax": [
                "Total Amount Before Tax"
            ]
        },
        "table_structure": {
                "columns": {
                    "Sr no": ["SR.", "SR No.", "No."],
                    "Name of Product/Service": ["Name of Product/Service", "Product/Service     Description", "Item Name", "Item Description", "Service/Product Title", "Product"],
                    "Product Code": ["Product Code", "Product ID", "Item Code", "SKU"],
                    "HSN/SAC Code": ["HSN/SAC Code", "HSN Code", "SAC Code", "HSN/SAC", "HSN Code"],
                    "QTY.": ["QTY", "Quantity", "Item Quantity", "Qry."], 
                    "UOM": ["UOM", "Unit of Measure", "Measurement Unit", "Unit"],
                    "Rate": ["Rate", "Unit Price", "Price per Unit"],
                    "Amount": ["Amount", "Total Amount", "Invoice Amount"],
                    "Discount": ["Discount", "Price Reduction"],
                    "Taxable Value": ["Taxable Value", "Taxable Amount"],
                    "Value": ["Value", "Total Value", "Net Value"]
               }
        },

     "coordinates": {
         "Date" : {
                "x1": 252.0,
                "y1": 155.14999389648438,
                "x2": 44.242431640625,
                "y2": 9.0
        },
        "LR No" : {
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": ""
        },
        "CIN":{
            "x1":102.0,
            "y1":137.19998168945312,
            "x2":93.8265380859375,
            "y2": 9.0
        },
        "GST NO":{
            "x1": 102.0,
            "y1": 113.14998626708984,
            "x2": 68.75135803222656,
            "y2": 9.0
        },
        "PAN No":{
            "x1": 102.0,
            "y1": 125.14999389648438,
            "x2": 43.909637451171875,
            "y2": 9.0
        },
        "LR Date":{
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": ""
        },
        "Round Off":{
            "x1": 567.5032348632812,
            "y1": 580.0999755859375,
            "x2": 14.53729248046875,
            "y2": 8.20001220703125
        },
        "Vehicle No.":{
            "x1": "",
            "y1": "",
            "x2": "",
            "y2":""
        },
        "invoice_date":{
            "x1": 252.0,
            "y1": 155.14999389648438,
            "x2": 44.242431640625,
            "y2":9.0
        },
        "Date Of Supply":{
            "x1": 432.0,
            "y1": 167.14999389648438,
            "x2":43.450439453125,
            "y2":9.0
        },
        "Party P.O. Ref":{
            "x1": 108.0,
            "y1": 167.14999389648438,
            "x2": 33.99757385253906,
            "y2":9.0
        },
        "Taxable Amount": {
            "x1": 553.0032348632812,
            "y1": 543.1499633789062,
            "x2": 29.05950927734375,
            "y2":8.20001220703125
        },
        "invoice_number": {
            "x1": 110.03399658203125,
            "y1": 155.14999389648438,
            "x2": 41.0655517578125,
            "y2": 9.0
        },
        "Add : Transport":{
            "x1": 563.3532104492188,
            "y1": 525.1499633789062,
            "x2": 18.6947021484375,
            "y2": 8.20001220703125
        },
        "Transporter":{
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": ""
        },
        "Transportation Mode":{
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": ""
        } ,
        "Total Amount After Tax": {
            "x1": 548.8532104492188,
            "y1": 599.0999755859375,
            "x2": 33.2169189453125,
            "y2": 8.20001220703125
        } ,
        "Total Amount Before Tax": {
            "x1": 552.80322265625,
            "y1": 507.1499938964844,
            "x2": 29.24810791015625,
            "y2": 8.199981689453125
        },
        "table_structure": {
                "columns":{
                    "Sr no": {
                        "x1": "",
                        "y1": "",
                        "x2": "",
                        "y2": ""
                    },
                    "Name of Product/Service": {
                        "x1": 38.0,
                        "y1": 404.08746337890625,
                        "x2": 99.67637634277344,
                        "y2": 7.350006103515625
                    },
                    "Product Code":{
                        "x1":186.74998474121094,
                        "y1":404.83746337890625,
                        "x2":33.16937255859375,
                        "y2":7.350006103515625
                   },
                   "HSN/SAC Code":{
                        "x1":232.4499969482422,
                        "y1":404.08746337890625,
                        "x2":29.810440063476562,
                        "y2":7.350006103515625
                    },
                    "QTY.":{
                        "x1":282.06109619140625,
                        "y1":404.08746337890625,
                        "x2":16.75677490234375,
                        "y2":7.350006103515625
                    },
                    "UOM":{
                        "x1":318.5,
                        "y1":404.08746337890625,
                        "x2":12.990936279296875,
                        "y2":7.350006103515625
                    },
                    "Rate":{
                        "x1":399.6166076660156,
                        "y1":404.08746337890625,
                        "x2":26.047149658203125,
                        "y2":7.350006103515625
                    },
                    "Amount":{
                        "x1":399.6166076660156,
                        "y1":404.08746337890625,
                        "x2":26.047149658203125,
                        "y2":7.350006103515625
                    },
                    "Discount":{
                        "x1":449.5111083984375,
                        "y1":404.08746337890625,
                        "x2":13.03033447265625,
                        "y2":7.350006103515625
                    },
                    "Taxable Value":{
                        "x1":399.6166076660156,
                        "y1":404.08746337890625,
                        "x2":26.047149658203125,
                        "y2":7.350006103515625
                    },
                    "Value":{
                        "x1":554.7664184570312,
                        "y1":404.08746337890625,
                        "x2":26.047119140625,
                        "y2":7.350006103515625
                      }    
                }
        }
    }

 },

"template_2": {                             # invoice_7
    "keys": {
        "Date": [
            "Date",
            "date",
            "Dated"
        ],
        "Phone": [
            "Phone",
            "Contact No",
            "Phone No",
            "Contact"
        ],
        "E-Mail": [
            "E-mail ID",
            "E-Mail",
            "email",
            "E-mail",
            "Email",
            "Email ID",
            "Ethail"
        ],
        "LR No": [
                "LR No.","LR No"
            ],
        "Booking": [
            "Booking",
            "booking"
        ],
        "Factory": [
            "Factory"
        ],
        "Freight": [
            "FREIGHT",
            "Freight",
            "FREIGHT IMP/EXP-AIR",
            "Freight Terms"
        ],
        "IRN No.": [
            "IRN NO.",
            "IRN Number",
            "IRN No",
            "IRN No.",
            "IRN"
        ],
        "LR Date": [
            "LR date",
            "LR Date"
        ],    
        "Website": [
            "Website","website"
        ],
        "Currency": [
            "Currency",
            "currency"
        ],
        "Sub Total": [
            "Sub Total"
        ],
        "Agent Code": [
                "Agent Code",
                "agentcode",
                "agent code"
        ],
        "Total Amount": [
            "Grand Total",
            "Total Amount",
            "Total",
            "Total Payble Amount",
            "Net Amount",
            "Basic Amount",
            "Tax Amount"
        ],
        "Transporter": [
            "Transporter",
            "Transporter Name"
        ],
        "Vehicle No.": [
            "Vehicle No",
            "Vehicle Number",
            "Vehicle No.",
            "Motor Vehicle No."
        ],
        "Regd. Office": [
            "Regd. Office"
        ],
        "invoice_date": [
            "Invoice date",
            "Invoice Date"
        ],
        "Payment Terms": [
            "Payment Terms",
            "Mode/Terms of Payment"
        ],
        "invoice_number": [
            "Invoice #",
            "Invoice No.",
            "Invoice No",
            "Invoice Number"
        ],
        "Buyer's Order No.": [
            "Buyer's Order No."
        ],
        "Destination": [
            "Final Destination","Destination", "Final"
        ],
        "Acknowledgement No.": [
            "Acknowledgement No",
            "Acknowledge Number",
            "Ack. No",
            "Ack. Number"
        ],
        "Acknowledgement Date": [
            "Ack. Date",
            "Acknowledge Date"
        ]
    },
        "table_structure": {
            "columns": {
                "Sr No": ["SR.", "SR No.","No."],
                "Item Description": ["Item Description"],
                "HSN/SAC Code": ["HSN/SAC Code","HSN No.", "HSN Code", "SAC Code", "Harmonized System of Nomenclature (HSN)", "Service Accounting Code (SAC)", "Tax Code", "Product Classification Code", "Service Classification Code", "HS Code", "Customs Code", "Item Tax Code", "Classification Code", "Goods/Service Code", "HSN/SAC Classification", "HSN/SAC Number"],
                "Batch No.": ["Batch No.", "Batch Number", "Lot No.", "Lot Number", "Manufacturing Batch", "Production Batch"],
                "No. of Pckgs": ["Number of Packages.", "Package Count", "No. of Boxes", "Package Units"],
                "Avg. Cont. in Kg.": ["Average Weight", "Avg Weight", "Average Content", "Avg. Weight per Package","Average Kg per Unit"],
                "Total Qty in Kgs": ["Total Weight", "Total Quantity", "Total Kg", "Total Weight in Kilograms","Total Weight (in Kgs)"],
                "Unit Price/Kg.": ["Price per Kg", "Unit Price (Kg)", "Cost per Kg", "Price per Kilogram","Kg Unit Price"],
                "Taxable value": ["Taxable Amount", "Value for Tax", "Taxable Base", "Taxable Price","Amount Subject to Tax"],
        }
    },
    "coordinates": {
        "Date":{
            "x1":473.4532470703125,
            "y1":182.80007934570312,
            "x2":39.58831787109375,
            "y2":8.199996948242188
        },
        "Phone":{
            "x1":164.86111450195312,
            "y1":141.08750915527344,
            "x2":90.15162658691406,
            "y2":7.3499908447265625
        },
        "E-Mail":{
            "x1":282.0500183105469,
            "y1":141.08750915527344,
            "x2":78.18392944335938,
            "y2":7.3499908447265625
        },
        "LR No":{
            "x1":421.357421875,
            "y1":339.1000061035156,
            "x2":4.15606689453125,
            "y2":8.199981689453125
        },
        "Booking":{
            "x1":123.45320129394531,
            "y1":295.1000061035156,
            "x2":41.15162658691406,
            "y2":8.199981689453125
        },
        "Factory":{
            "x1":124.6500473022461,
            "y1":130.28750610351562,
            "x2":383.55741119384766,
            "y2":7.3499908447265625
        },
        "Freight":{
            "x1":123.44999694824219,
            "y1":309.6000061035156,
            "x2":20.907028198242188,
            "y2":8.199981689453125
        },
        "IRN No.":{
            "x1":142.35000610351562,
            "y1":352.5,
            "x2":257.6181335449219,
            "y2":8.199981689453125
        },
        "LR Date":{
            "x1":"",
            "y1":"",
            "x2":"",
            "y2":""
        },
        "Website":{
            "x1":392.1500244140625,
            "y1":141.08750915527344,
            "x2":73.00244140625,
            "y2":7.3499908447265625
        },
        "Currency":{
            "x1":"",
            "y1":"",
            "x2":"",
            "y2":""
        },
        "Sub Total":{
            "x1":532.4531860351562,
            "y1":500.0500183105469,
            "x2":33.2169189453125,
            "y2":8.199981689453125
        },
        "Agent Code":{
            "x1":124.70320129394531,
            "y1":280.6000061035156,
            "x2":8.313446044921875,
            "y2":8.199981689453125
        },
        "Total Amount":{
            "x1":531.3532104492188,
            "y1":547.199951171875,
            "x2":31.2899169921875,
            "y2":8.20001220703125
        },
        "Transporter":{
            "x1":123.44999694824219,
            "y1":339.1000061035156,
            "x2":92.25387573242188,
            "y2":8.199981689453125
        },
        "Vehicle No.":{
            "x1":417.2032165527344,
            "y1":309.6000061035156,
            "x2":49.43426513671875,
            "y2":8.199981689453125
        },
        "Regd. Office":{
            "x1":"",
            "y1":"",
            "x2":"",
            "y2":""
        },
        "invoice_date":{
            "x1":473.4532470703125,
            "y1":182.80007934570312,
            "x2":39.58831787109375,
            "y2":8.199996948242188
        },
        "Payment Terms":{
            "x1":417.20001220703125,
            "y1":339.1000061035156,
            "x2":40.7459716796875,
            "y2":8.199981689453125
        },
        "invoice_number":{
            "x1":474.29473876953125,
            "y1":168.9625701904297,
            "x2":61.36383056640625,
            "y2":9.849990844726562
        },
        "Buyer's Order No.":{
            "x1":"",
            "y1":"",
            "x2":"",
            "y2":""
        },
        "Destination":{
            "x1":"",
            "y1":"",
            "x2":"",
            "y2":""
        },
        "Acknowledgement No.":{
            "x1":"",
            "y1":"",
            "x2":"",
            "y2":""
        },
        "table_structure": {
            "columns": {
                "Sr No":{
                    "x1":"",
                    "y1":"",
                    "x2":"",
                    "y2":""
                },
                "Item Description":{
                    "x1":62.84999084472656,
                    "y1":416.5,
                    "x2":63.64707946777344,
                    "y2":8.199981689453125
                },
                "HSN/SAC Code":{
                    "x1":199.69998168945312,
                    "y1":416.0,
                    "x2":37.390625,
                    "y2":8.199981689453125
                },
                "Batch No.":{
                    "x1":251.78359985351562,
                    "y1":416.0,
                    "x2":38.276275634765625,
                    "y2":8.199981689453125
                },
                "No. of Pckgs":{
                    "x1":306.5032043457031,
                    "y1":416.5,
                    "x2":14.53729248046875,
                    "y2":8.199981689453125
                },
                "Avg. Cont. in Kg.":{
                    "x1":352.30322265625,
                    "y1":415.0,
                    "x2":18.6947021484375,
                    "y2":8.199981689453125
                },
                "Total Qty in Kgs":{
                    "x1":352.30322265625,
                    "y1":415.0,
                    "x2":18.6947021484375,
                    "y2":8.199981689453125
                },
                "Unit Price/Kg.":{
                    "x1":460.7532043457031,
                    "y1":415.75,
                    "x2":22.85211181640625,
                    "y2":8.199981689453125
                },
                "Taxable value":{
                    "x1":532.4031982421875,
                    "y1":417.0,
                    "x2":31.16693115234375,
                    "y2":8.199981689453125
                }
            }
        }
    }
},

"template_3": {                      # invoice_9
    "keys": {
       "MSME": [
                "MSME",
                "Msme",
                "msme"
            ],
        "Phone": [
                "Contact No",
                "Phone No",
                "Phone",
                "Contact"
            ],
        "State": [
                "State",
                "state",
                "STATE"
            ],
        "CIN": [
                "CIN No",
                "CIN Number",
                "cin number",
                "Cin number",
                "CIN NO",
                "CINNo.",
                "CINNo"
            ],  
        "E-Mail": [
                "Email",
                "E-mail ID",
                "email"
            ],
        "GST NO": [
            "GST No",
            "GST NO", 
            "GSTIN Number", 
            "GSTIN Numbe", 
            "GSTTIN/UIN", 
            "GSTIN No",
            "GSTIN"
        ],
        "Round Off": [
            "Round Off",
            "Round off"      
        ],
        "Vehicle No.": [
            "Vehicle No",
            "Vehicle No.", 
            "Vehicle Number",
            "Veh.No.",
            "Motor Vehicle No."
        ],
        "invoice_date": [
            "Invoice Date", 
            "Dated"
        ],
        "Date Of Supply": [
                "Date Of Supply",
                "date_of_supply",
                "Supply date"
            ],
        "invoice_number": [
            "Invoice No", 
            "Invoice Number", 
            "Invoice #", 
            "Inv No."
        ],
        "Reverse Charge": [
                "reverse charge",
                "Reverse Charge"
            ],
        "place_of_supply": [
                "Place of Supply", 
                "Supply Place", 
                "Place Of Suppy"
            ],                  
            "Freight": [
              "FREIGHT", "Freight","FREIGHT IMP/EXP-AIR","Freight Terms"
            ],
            "Value": [
                "Total Payable Amount","Net Amount"
            ],
            "Transportation Mode": [
                "Transportation Mode"
            ],
            "Total Amount Before Tax": [
                "Total Amount Before Tax" , "Sub Total"
            ]
    },
            "table_structure": {
                "columns": {
                   "Sr No": ["SR.", "SR No.","No."],
                   "Item Description": ["Item Description", "Product Description", "Service Description", "Description", "Item Details", "Product Details"],
                   "HSN/SAC Code": ["HSN/SAC Code","HSN No.", "HSN Code", "SAC Code", "Harmonized System of Nomenclature (HSN)", "Service Accounting Code (SAC)", "Tax Code", "Product Classification Code", "Service Classification Code", "HS Code", "Customs Code", "Item Tax Code", "Classification Code", "Goods/Service Code", "HSN/SAC Classification", "HSN/SAC Number"],
                    "GST Rate %": ["GST Percentage", "GST Rate", "GST (%)", "GST % on Product"],
                    "QTY.": ["QTY", "Quantity", "Item Quantity", "Product Quantity", "Units", "No. of Items", "Total Quantity", "Ordered Quantity", "Quantity Ordered", "Quantity of Product", "Item Count", "Units Ordered", "Quantity of Items", "Product Units"],
                    "UOM": ["UOM", "Unit of Measure", "Measurement Unit", "Unit", "Unit Type", "Measurement Type", "Unit Quantity", "Product Unit", "Service Unit", "Quantity Unit", "Measurement Unit Code", "Product Measurement Unit", "Service Measurement Unit", "UOM Code"],
                    "Rate": ["Rate", "Unit Price", "Price per Unit", "Item Rate", "Product Rate", "Cost per Unit"],
                    "Discount": ["Discount", "Price Reduction", "Discounted Price", "Total Discount", "Discount Amount", "Rebate"],
                    "Amount": ["Amount", "Total Amount", "Invoice Amount", "Net Amount", "Total Price", "Amount Due"],
        }
    },
    "coordinates" : {
        "MSME":{
            "x1":101.90003967285156,
            "y1":146.10000610351562,
            "x2":78.03280639648438,
            "y2":8.199996948242188
        },
        "Phone":{
            "x1":101.90084075927734,
            "y1":112.3499984741211,
            "x2":100.38297271728516,
            "y2":8.200004577636719
        },
        "State":{
                "x1":143.89999389648438,
                "y1":199.6875,
                "x2":22.123199462890625,
                "y2":7.3499908447265625
        },
        "CIN":{
                "x1":101.95003509521484,
                "y1":134.85000610351562,
                "x2":88.88562774658203,
                "y2":8.199996948242188
        },
        "E-Mail":{
                "x1":101.90003204345703,
                "y1":101.0999984741211,
                "x2":94.2411880493164,
                "y2":8.200004577636719
        },
        "GST NO":{
                "x1":101.95003509521484,
                "y1":123.5999984741211,
                "x2":62.42462921142578,
                "y2":8.200004577636719
        },
        "Round Off":{
                "x1":551.4990844726562,
                "y1":431.23748779296875,
                "x2":13.0302734375,
                "y2":7.350006103515625
        },
        "Vehicle No.":{
            "x1":"",
            "y1":"",
            "x2":"",
            "y2":""
        },
        "invoice_date":{
                "x1":143.91110229492188,
                "y1":189.1875,
                "x2":35.48463439941406,
                "y2":7.3499908447265625
        },
        "Date Of Supply":{
            "x1":403.3000183105469,
            "y1":188.4375,
            "x2":35.484527587890625,
            "y2":7.3499908447265625
        },
        "invoice_number":{
                "x1":143.89999389648438,
                "y1":178.0500030517578,
                "x2":12.470840454101562,
                "y2":8.199996948242188
        },
        "Reverse Charge":{
            "x1":143.89999389648438,
            "y1":167.4375,
            "x2":9.614608764648438,
            "y2":7.3499908447265625
        },
        "place_of_supply":{
                "x1":"",
                "y1":"",
                "x2":"",
                "y2":""
        },
        "Freight":{
                "x1":539.7302856445312,
                "y1":406.6374816894531,
                "x2":24.8955078125,
                "y2":6.550018310546875
        },
        "Value":{
                "x1":520.0499877929688,
                "y1":442.23748779296875,
                "x2":44.37799072265625,
                "y2":7.350006103515625
        },
        "Transportation Mode":{
                "x1": 403.3000183105469,
                "y1": 167.4375,
                "x2": 17.394744873046875,
                "y2": 7.3499908447265625
        },
        "Total Amount Before Tax":{
                "x1": 532.7610473632812,
                "y1": 419.73748779296875,
                "x2": 31.6624755859375,
                "y2": 7.350006103515625
        },
        "table_structure": {
            "columns": {
                "Sr No":{
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": ""
                },
                "Item Description":{
                    "x1": 64.60006713867188,
                    "y1": 384.2874755859375,
                    "x2": 97.65177917480469,
                    "y2": 7.350006103515625
                },
                "HSN/SAC Code":{
                    "x1": 261.8500671386719,
                    "y1": 383.9374694824219,
                    "x2": 29.86700439453125,
                    "y2": 6.550018310546875
                },
                "GST Rate %":{
                    "x1": 280.10498046875,
                    "y1": 383.9374694824219,
                    "x2": 11.612091064453125,
                    "y2": 6.550018310546875
                },
                "QTY.":{
                    "x1": 353.1611022949219,
                    "y1": 384.2874755859375,
                    "x2": 16.75677490234375,
                    "y2": 7.350006103515625
                },
                "UOM":{
                    "x1": 389.1000061035156,
                    "y1": 384.2874755859375,
                    "x2": 12.990936279296875,
                    "y2": 7.350006103515625
                },
                "Rate": {
                    "x1": 532.7610473632812,
                    "y1": 384.2874755859375,
                    "x2": 31.6624755859375,
                    "y2": 7.350006103515625
                },
                "Discount": {
                    "x1": 280.10498046875,
                    "y1": 383.9374694824219,
                    "x2": 11.612091064453125,
                    "y2": 6.550018310546875
                },
                "Amount": {
                    "x1": 532.7610473632812,
                    "y1": 384.2874755859375,
                    "x2": 31.6624755859375,
                    "y2": 7.350006103515625
                }
            }
        }
    }
},

"template_4": {                       
    "keys": {
        "MSME": [
            "MSME NO",
            "MSME",
            "Msme",
            "msme"
        ],
        "PR No": [
            "PR NO"
        ],
        "GST NO": [
            "GST No",
             "GST NO", 
            "GSTIN Number", 
            "GSTIN Numbe", 
            "GSTTIN/UIN", 
            "GSTIN No",
            "GSTIN NO" ,
            "GSTIN"
        ],
        "PAN No": [
            "PAN Number",
            "PAN No",
            "PAN NO"
        ],
        "Phone": [
            "Contact No",
            "Phone No",
            "Phone",
            "Contact",
            "Mobile",
            "Mob.",
            "Mob"
        ],
        "E-Mail": [
            "E-mail ID",
            "email",
            "E-Mail",
            "E-mail",
            "Email"
        ],
        "Factory address": [
                "Factory Add"
        ],
        "vendor_name": [
            "VENDOR NAME"
        ],
        "Office address": [
            "Office Add"
        ],
        "Purchase Order": [
            "PURCHASE ORDER"
        ],
        "Vendor Address": [
            "VENDOR ADDRESS"
        ],
        "Purchase PO Date": [
                "PURCHASE PO DATE"
        ],
        "Vendor contact no": [
            "VENDOR CONTACT NO"
        ]
    },   
        "table_structure": {
            "columns": {
                "Sr No":  ["SR.", "SR No.","No."],
                "Product Code": ["Product Code", "Item Code", "Product ID", "Item ID", "Catalog Number", "Item Number", "Stock Code", "Inventory Code", "Model Number", "Reference Code", "Product Reference"],
                "Item Description": ["Item Description", "Product Description", "Service Description", "Description", "Item Details", "Product Details"],
                "QTY.": ["QTY", "Quantity", "Item Quantity", "Product Quantity", "Units", "No. of Items", "Total Quantity", "Ordered Quantity", "Quantity Ordered", "Quantity of Product", "Item Count", "Units Ordered", "Quantity of Items", "Product Units"],
                "Rate": ["Rate", "Unit Price", "Price per Unit", "Item Rate", "Product Rate", "Cost per Unit"],
                "Currency": ["Currency","currency"],
                "Amount": ["Amount", "Total Amount", "Invoice Amount", "Net Amount", "Total Price", "Amount Due"],

        }   
        },  
    "coordinates":{
        "MSME": {
                "x1": "",
                "y2": "",
                "x2": "",
                "y2": ""
            },
        "PR No": {
                "x1": "",
                "y2": "",
                "x2": "",
                "y2": ""
            },
        "GST NO": {
            "x1": 52.60000228881836,
            "y1": 160.35000610351562,
            "x2": 63.03207015991211,
            "y2": 8.199996948242188
        },
        "PAN No": {
            "x1": 188.09999084472656,
            "y1": 160.35000610351562,
            "x2": 42.56591796875,
            "y2": 8.199996948242188
        },
        "Phone": {
            "x1": 67.59999084472656,
            "y1": 173.85000610351562,
            "x2": 41.572669982910156,
            "y2": 8.199996948242188
        },
        "E-Mail": {
            "x1": 191.59999084472656,
            "y1": 173.85000610351562,
            "x2": 78.94361877441406,
            "y2": 8.199996948242188
        },
        "Factory address": {
            "x1": 68.69999694824219,
            "y1": 113.0999984741211,
            "x2": 203.4830780029297,
            "y2": 8.200004577636719
        },
        "vendor_name": {
                "x1": 147.0,
                "y1": 190.85000610351562,
                "x2": 100.71125793457031,
                "y2": 8.199996948242188
        },
        "Office address": {
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": ""
        },
        "Purchase Order": {
            "x1": 438.75,
            "y1": 190.35000610351562,
            "x2": 74.60235595703125,
            "y2": 8.199996948242188
        },
        "Vendor Address": {
                "x1": 147.0,
                "y1": 206.85000610351562,
                "x2": 54.484161376953125,
                "y2": 8.199996948242188
        },
        "Purchase PO Date": {
                "x1": 438.75,
                "y1": 207.35000610351562,
                "x2": 39.58831787109375,
                "y2": 8.199996948242188
        },
        "Vendor contact no": {
            "x1":  438.75,
            "y1": 246.35000610351562,
            "x2": 45.73016357421875,
            "y2": 8.199996948242188
        },
        "table_structure": {
            "columns": {
                "Sr No": {
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": ""
                },
                "Product Code": {
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": ""
                },
                "Item Description": {
                    "x1": 119.99993896484375,
                    "y1": 315.48748779296875,
                    "x2": 65.57568359375,
                    "y2": 7.350006103515625
                },
                "QTY.": {
                    "x1": 245.46104431152344,
                    "y1": 313.5374755859375,
                    "x2": 24.209701538085938,
                    "y2": 7.350006103515625
                },
                "Rate": {
                    "x1": 370.4110412597656,
                    "y1": 313.5374755859375,
                    "x2": 16.75677490234375,
                    "y2": 7.350006103515625
                },
                "Currency": {
                    "x1": 438.449951171875,
                    "y1": 313.5374755859375,
                    "x2": 11.87371826171875,
                    "y2": 7.350006103515625
                },
                "Amount": {
                    "x1": 508.5999450683594,
                    "y1": 313.3500061035156,
                    "x2": 37.374298095703125,
                    "y2": 8.199981689453125
                }

            }
        }
    }
},

"template_5": {
    "keys": {
        "Vehicle No.": [
                "Veh.No.",
                "Vehicle No",
                "Vehicle Number",
                "Vehicle No.",
            ],
        "Document": [
                "DOCUMENT"
            ],
        "GST NO":[
                "GSTIN NO.",
                "GST No",
                "GST NO", 
                "GSTIN Number", 
                "GSTIN Numbe", 
                "GSTTIN/UIN", 
                "GSTIN No",
                "GSTIN NO",
                "GSTIN"           # added 
            ],
        "Challan No":[
                "Challan No"
            ],
        "Document Date": [
                "DOCUMENT DATE"
            ],
        "Eway Bill No.": [
                "Eway Bill No",
                "e-Way Bill No.",
                "Eway Bill No.",
                "E-way Bill No",
                "EWay Bil No",
                "EWay Bil No."
            ],
        "Party Bill No":[
                "Party Bill No"
            ],
        "Removal Date Time": [
                "Removal Date Time"
            ],
    },
    "table_structure": {
        "columns": {
            "Sr no": ["SR.", "SR No.","No."],
            "Item Description": ["Item Description", "Product Description", "Service Description", "Description", "Item Details", "Product Details"],
            "HSN/SAC Code":["HSN/SAC Code","HSN No.", "HSN Code", "SAC Code", "Harmonized System of Nomenclature (HSN)", "Service Accounting Code (SAC)", "Tax Code", "Product Classification Code", "Service Classification Code", "HS Code", "Customs Code", "Item Tax Code", "Classification Code", "Goods/Service Code", "HSN/SAC Classification", "HSN/SAC Number"],
            "QTY.":["QTY", "Quantity", "Item Quantity", "Product Quantity", "Units","Unit", "No. of Items", "Total Quantity", "Ordered Quantity", "Quantity Ordered", "Quantity of Product", "Item Count", "Units Ordered", "Quantity of Items", "Product Units"],
            "Rate":["Rate", "Unit Price", "Price per Unit", "Item Rate", "Product Rate", "Cost per Unit"],
            "Discount": ["Discount", "Price Reduction", "Discounted Price", "Total Discount", "Discount Amount", "Rebate"],
            "S.Gst %": ["State GST Rate", "SGST Percentage", "SGST (%)", "State GST %","SGST Rate (%)"],
            "C.Gst %": ["Central GST Rate", "CGST Percentage", "CGST (%)", "Central Tax Rate","CGST Rate (%)"],
            "Taxable value": ["Taxable Amount", "Value for Tax", "Taxable Base", "Taxable Price","Amount Subject to Tax"]
        }
    }
},

"template_6": {               # invoice_12
    "keys": {
        "Address":["Adress","Address","address","Adres5"],
        "Phone": ["Contact No", "Phone", "Phone No", "Contact","Mobile","Mob.","Mob","Phone :","Phone:","Cell No.","Cell No"],
        "E-Mail": [
            "E-mail ID",
            "E-Mail",
            "email",
            "E-mail",
            "Email",
            "Email ID",
            "Ethail"
        ],
        "CIN": [
                "CIN No",
                "CIN Number",
                "cin number",
                "Cin number",
                "CIN NO",
                "CINNo.",
                "CINNo"
            ],  
       "State": ["State", "state", "STATE","State Name"],
       "GST NO": [
        "GST No",
        "GST NO",
        "GSTIN Number",
        "GSTIN Numbe",
        "GSTTIN/UIN",
        "GSTIN No",
        "GSTIN",
        "GSTIN:"
      ],
       "PO No.":["PO No.","PO No","PO Number","PO No :","PO No:","PO.No."],
        "invoice_number": [
        "Invoice No",
        "Invoice Number",
        "Invoice #",
        "Inv No.",
        "invoice_number",
        "Invoice No.",# changes
        "invoice No." ,# chnages
        "nvoice No.",
        "nvoice No"
      ],
     "invoice_date": ["Invoice Date", "Dated","invoice_date","Invoice Date "],
     "Challan No": ["Challan No","Challan No."],
     "Eway Bill No.": ["Eway Bill No.","e-Way Bill No.","E-way Bill No","EWay Bil No","EWay Bil No.","e-Way Bil No.","e-Way Bil No"],
        "GST NO": [
                "GST No",
                "GST NO", 
                "GSTIN Number", 
                "GSTIN Numbe", 
                "GSTTIN/UIN", 
                "GSTIN No",
                "GSTIN NO" ,           # added 
                "GSTIN NO.",
                "GSTIN"
            ],
       "Acknowledgement No.": [
        "Acknowledgement No",
        "Acknowledge Number",
        "Ack. No",
        "Ack. Number",
        "Acknowledgement No.",
        "Ack No."           # changes
      ],
      "Date": ["Date", "date","Dated"], 
      "IRN No.": ["IRN No.", "IRN No", "IRN Number", "IRN NO.","IRN"]
    },
    "table_structure": {
        "columns": {
            "Item Description": ["Item Description", "Product Description", "Service Description", "Description", "Item Details", "Product Details","Particulars"],
            "HSN/SAC Code": ["HSN/SAC Code","HSN No.", "HSN Code", "SAC Code", "Harmonized System of Nomenclature (HSN)", "Service Accounting Code (SAC)", "Tax Code", "Product Classification Code", "Service Classification Code", "HS Code", "Customs Code", "Item Tax Code", "Classification Code", "Goods/Service Code", "HSN/SAC Classification", "HSN/SAC Number"],
            "Currency": ["Currency","currency"],
            "QTY.": ["QTY", "Quantity", "Item Quantity", "Product Quantity", "Units","Unit", "No. of Items", "Total Quantity", "Ordered Quantity", "Quantity Ordered", "Quantity of Product", "Item Count", "Units Ordered", "Quantity of Items", "Product Units"],
            "Rate": ["Rate", "Unit Price", "Price per Unit", "Item Rate", "Product Rate", "Cost per Unit"],
            "Amount": ["Amount", "Total Amount", "Invoice Amount", "Net Amount", "Total Price", "Amount Due"],
            "GST Rate %": ["GST Percentage", "GST Rate", "GST (%)", "GST % on Product"],
            "IGST": [
                    "IGST",
                    "Integrated GST",
                    "GST on Interstate Supply",
                    "Interstate GST",
                    "IGST Tax",
                    "IGST Amount",
                    "GST on Import"
                ],
            "Amount": ["Amount", "Total Amount", "Invoice Amount", "Net Amount", "Total Price", "Amount Due"]
        }
    }
},

  "template_7": {
    "keys": {
        "C.Gst %": [
                "Central GST Rate", "CGST Percentage", "CGST (%)", "Central Tax Rate","CGST Rate (%)","CGST"
            ],
        "S.Gst %": [
                "State GST Rate", "SGST Percentage", "SGST (%)", "State GST %","SGST Rate (%)","SGST"
            ],
        "LR No": [
                "LR No.",
                "LR no",
                "LR No"
            ],
        "PAN No": [
                "PAN NO.",
                "PAN No",
                "PAN Number"
            ],
        "Phone": [
                "Contact",
                "Phone",
                "Contact No",
                "Phone No",
            ],
        "IRN No.": [
                "IRN No",
                "IRN NO.",
                "IRN Number",
                "IRN No.",
                "IRN"
            ],
        "Acknowledgement Date":[
                "Ack Date",
                "Ack. Date",
                "Acknowledge Date"
            ],
        "Round Off": [
                "Round Off"
            ],
        "State Code": [
                "State-Code",
                "State Code",
                "StateCode",
               "statecode",
                "state-code",
                "stateCode",
                "Statecode",
                "state code"
            ],
        "Sub Total": [
                "Sub Total"
            ],
        "Receiver Details": [
                "Buyer's Details","Receiver's Details","Buyer Name","Contact Person","Consignee Details","Bill To","Buyer (Bill to)"
            ],
                "Total Amount": [
                "Total Amount",
                "Grand Total",
                "Total",
                "Total Payble Amount",
                "Net Amount",
                "Basic Amount",
                "Tax Amount",
                "Grand Total INR"
            ],
        "Acknowledgement No.": [
                "Ack No","Acknowledgement No.","Ack No."
            ],
        "Buyer GST NO": [
                "Buyer GSTIN No."
            ],
        "Vehicle No.": [
                "Vehicle No",
                "Vehicle Number",
                "Vehicle No."
            ],
        "GST NO": [
                "GSTIN No.",
                "GST No",
                "GST NO",
                "GSTIN Number",
                "GSTIN Numbe",
                "GSTTIN/UIN",
                "GSTIN No",
                "GSTIN NO" ,                      
                "GSTIN"                   
            ],
        "invoice_date": [
                "Dated"
            ],
        "Payment Mode": [
                "Payment Mode",
                "Mode/Terms of Payment",
                "mode_of_payment"
            ],
        "Payment Terms": [
                "Payment Terms",
                "Mode/Terms of Payment"
            ],
        "invoice_number": [
                "Invoice",
                "Invoice No",
                "Invoice No.",
                "invoice No."
            ],
        "Order Ref. No.": [
                "Order Ref. No"
            ],
            
        "Taxable Amount": [
                "Taxable Amount"
            ]
    },

    "table_structure": {
        "columns": {
            "Sr no": ["SR.", "SR No.","No."],
            "Item Description": ["Item Description", "Product Description", "Service Description", "Description", "Item Details", "Product Details","Particulars"],
            "HSN/SAC Code": ["HSN/SAC Code","HSN No.", "HSN Code", "SAC Code", "Harmonized System of Nomenclature (HSN)", "Service Accounting Code (SAC)", "Tax Code", "Product Classification Code", "Service Classification Code", "HS Code", "Customs Code", "Item Tax Code", "Classification Code", "Goods/Service Code", "HSN/SAC Classification", "HSN/SAC Number"],
            "QTY.": ["QTY", "Quantity", "Item Quantity", "Product Quantity", "No. of Items", "Total Quantity", "Ordered Quantity", "Quantity Ordered", "Quantity of Product", "Item Count", "Units Ordered", "Quantity of Items", "Product Units"],
            "Rate with Tax": ["Tax Inclusive Price", "Price with Tax", "Total Price (Including Tax)", "Price After Tax","Amount with Tax"],
            "Rate": ["Rate", "Unit Price", "Price per Unit", "Item Rate", "Product Rate", "Cost per Unit","Basic Rate"],
            "Discount": ["Discount", "Price Reduction", "Discounted Price", "Total Discount", "Discount Amount", "Rebate"],
            "Amount": ["Amount", "Total Amount", "Invoice Amount", "Net Amount", "Total Price", "Amount Due"]
        }
    }
},
  "template_8": {             # invoice_11
    "keys": {
        "Phone": [
                "Contact No", "Phone", "Phone No","Contact","Mobile","Mob.","Mob"
            ],
        "GST NO": [
                "GST No",
                "GST NO",
                "GSTIN Number",
                "GSTIN Numbe",
                "GSTTIN/UIN",
                "GSTIN No",
                "GSTIN NO" ,                      
                "GSTIN No."   ,
                "GSTIN"                   
            ],
        "PAN No": [
                "PAN No", "PAN Number","PAN NO","PAN NO."
            ],
        "Receiver Details": [
                 "Buyer's Details",
                 "Receiver Details Billed to",
                 "Receiver Details",
                 "Billed to",
                 "Consignee Details",
                 "Buyer Name",
                 "Consignee Details",
                 "Bill To",
                 "Buyer (Bill to)"

            ],
        "IRN No.": [
                "IRN No.", "IRN No", "IRN Number", "IRN NO.","IRN"
            ],
        "Acknowledgement No.": [
                "Acknowledgement No",
                "Acknowledge Number",
                "Ack. No",
                "Ack. Number",
                "Acknowledgement No.",
                "Ack. No." 
            ],
        "Acknowledgement Date":[
                "Acknowledge Date", "Ack. Date"
            ],
        "invoice_number": [
                "Invoice No",
                "Invoice Number",
                "Invoice #",
                "Inv No.",
                "invoice_number",
                "Invoice No.",
                "invoice No."
            ],
        "Date": [
                "Date", "date","Dated"
            ],
        "Challan No": [
                "Challan No","Challan No."
            ],
        "Payment Mode": [
                "Payment Mode", "mode_of_payment","Mode/Terms of Payment"
            ],
        "Sale's Person": [
                "Sale's Person"
            ],
        "Delivery Through": [
                "Delivery Through","Dispatched through"
            ],
        "LR No": [
                "LR No", "LR no", "LR No.", "LR Number"
            ],
        "Vehicle No.": [
                "Vehicle No", "Vehicle No.", "Vehicle Number", "Veh.No.","Motor Vehicle No."
            ]
    },

    "table_structure": {
        "columns": {
            "Sr no": ["SR.", "SR No.","No.","Sr."],
            "Item Description": ["Item Description", "Product Description", "Service Description", "Description", "Item Details", "Product Details","Particulars","Description of Goods"],
            "HSN/SAC Code": ["HSN/SAC Code","HSN No.", "HSN Code", "SAC Code", "Harmonized System of Nomenclature (HSN)", "Service Accounting Code (SAC)", "Tax Code", "Product Classification Code", "Service Classification Code", "HS Code", "Customs Code", "Item Tax Code", "Classification Code", "Goods/Service Code", "HSN/SAC Classification", "HSN/SAC Number"],
            "GST Rate %": ["GST Percentage", "GST Rate", "GST (%)", "GST % on Product"],
            "QTY.": ["QTY", "Quantity", "Item Quantity", "Product Quantity", "No. of Items", "Total Quantity", "Ordered Quantity", "Quantity Ordered", "Quantity of Product", "Item Count", "Units Ordered", "Quantity of Items", "Product Units","Qty"],
            "UOM": ["UOM","Unit of Measure","Measurement Unit","Unit","Unit Type", "Measurement Type","Unit Quantity","Product Unit","Service Unit","Quantity Unit","Measurement Unit Code","Product Measurement Unit","Service Measurement Unit","UOM Code"],
            "Rate with Tax": ["Tax Inclusive Price", "Price with Tax", "Total Price (Including Tax)", "Price After Tax","Amount with Tax","Rate with Tax"],
            "Rate": ["Rate", "Unit Price", "Price per Unit", "Item Rate", "Product Rate", "Cost per Unit","Basic Rate"],
            "Discount": ["Discount", "Price Reduction", "Discounted Price", "Total Discount", "Discount Amount", "Rebate","Disc%"],
            "Amount": ["Amount", "Total Amount", "Invoice Amount", "Net Amount", "Total Price", "Amount Due"]
        }
    }
},

 "template_9": {                      # invoice_10
    "keys": {
        "GST NO": [
            "GST No",
            "GST NO",
            "GSTIN Number",
            "GSTIN Numbe",
            "GSTTIN/UIN",
            "GSTIN No",
            "GSTIN",
            "GSTIN:"
        ],
        "E-Mail": [
            "Email",
            "E-mail ID",
            "email",
            "E-Mail",
            "E-mail Id",
            "Email:",
            "email:"
        ],
        "Website": ["Website", "website"],
        "CIN": [
            "CIN No",
            "CIN Number",
            "cin number",
            "Cin number",
            "CIN NO",
            "CIN"
        ],
        "Purchase Order": [
            "PURCHASE ORDER",
            "Purchase order",
            "PURCHASE ORDER No",
            "Purchase Order No"
        ],
        "PO No.": [
            "PO No.",
            "PO No",
            "PO Number",
            "PO No :",
            "PO No:",
            "PO. No."
        ],
        "PO Date": ["PO Date", "PO Date:"],
        "Payment Terms": [
            "Payment Terms",
            "Mode/Terms of Payment",
            "Payment Terms:",
            "Payment Terms :"
        ],
        "Phone": [
            "Contact No",
            "Phone",
            "Phone No",
            "Contact",
            "Mobile",
            "Mob.",
            "Mob",
            "Phone :",
            "Phone:"
        ],
        "PAN No": [
            "PAN No",
            "PAN Number",
            "PAN NO",
            "PAN NO:"
        ],
        "Total Amount": [
            "Total Amount",
            "Grand Total",
            "Total",
            "Total Payble Amount",
            "Net Amount",
            "Basic Amount",
            "Tax Amount",
            "Grand Total INR",
            "Total Amount: INR",
            "Grand Total : INR"
        ],
        "Freight": ["FREIGHT", "Freight", "FREIGHT IMP/EXP-AIR", "Freight Terms"],
        "Insurance Terms": ["Insurance Terms"]
    },
    "table_structure": {
        "columns": {
            "Sr no": [
                "SR.",
                "SR No.",
                "No.",
                "Sr.",
                "Sr No"
            ],
            "Item Description": [
                "Item Description",
                "Product Description",
                "Service Description",
                "Description",
                "Item Details",
                "Product Details",
                "Particulars",
                "Description of Goods",
                "Item Desc"
            ]
        }
    }
},

 "template_10": {                          # invoice_13
    "keys": {
        "Order No":["Order No","Order No."],
        "PO Date": ["PO Date","PO Date:"],
        "Phone": ["Contact No", "Phone", "Phone No", "Contact","Mobile","Mob.","Mob","Phone :","Phone:","Cell No.","Cell No"],
        "E-Mail": ["Email", "E-mail ID", "email","E-Mail","E-mail Id", "Email:","email:","Email ID","Ethail"],
        "GST NO": [
        "GST No",
        "GST NO",
        "GSTIN Number",
        "GSTIN Numbe",
        "GSTTIN/UIN",
        "GSTIN No",
        "GSTIN",
        "GSTIN:",
        "ISTIN"
      ],
      "State": ["State", "state", "STATE","State Name"],
      "Location Code":["Location Code"]
    },

    "table_structure": {
        "columns": {
            "Sr no": ["SR.", "SR No.","No.","Sr."],
            "Item Description": ["Item Description", "Product Description", "Service Description", "Description", "Item Details", "Product Details","Particulars","Description of Goods"],
            "HSN/SAC Code": ["HSN/SAC Code","HSN No.", "HSN Code", "SAC Code", "Harmonized System of Nomenclature (HSN)", "Service Accounting Code (SAC)", "Tax Code", "Product Classification Code", "Service Classification Code", "HS Code", "Customs Code", "Item Tax Code", "Classification Code", "Goods/Service Code", "HSN/SAC Classification", "HSN/SAC Number"],
            "GST Rate %": ["GST Percentage", "GST Rate", "GST (%)", "GST % on Product"],
            "QTY.": ["QTY", "Quantity", "Item Quantity", "Product Quantity", "No. of Items", "Total Quantity", "Ordered Quantity", "Quantity Ordered", "Quantity of Product", "Item Count", "Units Ordered", "Quantity of Items", "Product Units","Qty"],
            "UOM": ["UOM","Unit of Measure","Measurement Unit","Unit","Unit Type", "Measurement Type","Unit Quantity","Product Unit","Service Unit","Quantity Unit","Measurement Unit Code","Product Measurement Unit","Service Measurement Unit","UOM Code"],
            "Rate with Tax": ["Tax Inclusive Price", "Price with Tax", "Total Price (Including Tax)", "Price After Tax","Amount with Tax","Rate with Tax"],
            "Rate": ["Rate", "Unit Price", "Price per Unit", "Item Rate", "Product Rate", "Cost per Unit","Basic Rate"],
            "Discount": ["Discount", "Price Reduction", "Discounted Price", "Total Discount", "Discount Amount", "Rebate","Disc%"],
            "Amount": ["Amount", "Total Amount", "Invoice Amount", "Net Amount", "Total Price", "Amount Due"]
        }
    }
},

 "template_11": {                             # invoice_15   
    "keys": {
        "Regd. Office": ["Regd. Office","REGD.ADD"],
        "BILL NO.": ["BILL NO","Bill No"],
        "Bank":["Bank"],
        "Account No":["Account No"],
        "Branch":["Branch"],
        "Bill Date": ["eWay Bill Date", "eWay Bill Valid Date","Bill Date","Bill Date >"],
        "Challan No": ["Challan No","Challan No."],
        "State": ["State", "state", "STATE","State Name"],
        "Order No":["Order No","Order No."],
        "E-Mail": ["Email", "E-mail ID", "email","E-Mail","E-mail Id", "Email:","email:","Email ID","E-mail Id"],
        "Eway Bill No.": ["Eway Bill No.","e-Way Bill No.","E-way Bill No","EWay Bil No","EWay Bil No.","e-Way Bil No.","e-Way Bil No"],
        "GST NO": [
        "GST No",
        "GST NO",
        "GSTIN Number",
        "GSTIN Numbe",
        "GSTTIN/UIN",
        "GSTIN No",
        "GSTIN",
        "GSTIN:"
      ],
      "State": ["State", "state", "STATE","State Name"],
      "Phone": ["Contact No", "Phone", "Phone No", "Contact","Mobile","Mob.","Mob","Phone :","Phone:","Cell No.","Cell No"],
      "E-Mail": [
            "E-mail ID",
            "E-Mail",
            "email",
            "E-mail",
            "Email",
            "Email ID",
            "Ethail"
        ],
    "Total Amount": [
        "Total Amount",
        "Grand Total",
        "Total",
        "Total Payble Amount",
        "Net Amount",
        "Basic Amount",
        "Tax Amount",
        "Grand Total INR",
        "Total Amount: INR",
        "Grand Total: INR",
        "Tax Amount(in words) INR"
    ],
        "PAN No": ["PAN No", "PAN Number","PAN NO","PAN NO:","PAN"],
    },
    "table_structure": {
        "columns": {
            "Sr no": [
                "SR.",
                "SR No.",
                "No.",
                "Sr.",
                "Sr No"
            ],
            "Item Description": [
                "Item Description",
                "Product Description",
                "Service Description",
                "Description",
                "Item Details",
                "Product Details",
                "Particulars",
                "Description of Goods",
                "Item Desc"
            ]
        }
    }
},
 
  "template_12": {                                        # invoice_14  
    "keys": {
     "Company Name":["Company Name","Company"],                      
      "Address":["Adress","Address","address"],
      "Contact Person":["Contact person","Contact Peraon"],
      "Receiver Details": [
        "Receiver Details Billed to",
        "Receiver Details",
        "Billed to",
        "Consignee Details",
        "Buyer Name",
        "Consignee Details",
        "Buyer's Details",
        "Bill To",
        "Buyer (Bill to)",
        "Bill  To",
        "Biling To",
        "Blling"
      ],
      "Billing Address":["Billing Address","Biling Address"],
      "Payment Terms": [
            "Payment Terms",
            "Mode/Terms of Payment"
        ],
      "Phone": ["Mobile No.","Mobile No"],
      "E-Mail": ["Email", "E-mail ID", "email","E-Mail","E-mail Id", "Email:","email:","Email ID","E-mail Id"],
      "GST NO": [
        "GST No",
        "GST NO",
        "GSTIN Number",
        "GSTIN Numbe",
        "GSTTIN/UIN",
        "GSTIN No",
        "GSTIN",
        "GSTIN:"
      ],
      "Freight": ["FREIGHT", "Freight", "FREIGHT IMP/EXP-AIR","Freight Terms"],
      "Document No":["Document No","Doc No.","Doc No"],
      "Packaging":["PACKAGING"],
      "Warranty":["Warranty","WARRANTY"]
    },
    "table_structure": {
        "columns": {
            "Sr no": [
                "SR.",
                "SR No.",
                "No.",
                "Sr.",
                "Sr No"
            ],
            "Item Description": [
                "Item Description",
                "Product Description",
                "Service Description",
                "Description",
                "Item Details",
                "Product Details",
                "Particulars",
                "Description of Goods",
                "Item Desc"
            ]
        }
    }
  },

 "template_13": {                                        
    "keys": {
     "Vendor code":["venso Code","Vendor Code"],                      
      "GST NO": [
        "GST No",
        "GST NO",
        "GSTIN Number",
        "GSTIN Numbe",
        "GSTTIN/UIN",
        "GSTIN No",
        "GSTIN",
        "GSTIN:",
        "GST No."
      ],
      "E-Mail": [
            "E-mail ID",
            "E-Mail",
            "email",
            "E-mail",
            "Email",
            "Email ID",
            "Ethail"
        ],
      "Company Name":["Company Name","Company","M.S"],  
      "PO No.":["PO No.","PO No","PO Number","PO No :","PO No:","P.O. No."],
      "PO Date": ["PO Date","PO Date:","P.O. Date"], 
      "Payment Terms": [
            "Payment Terms",
            "Mode/Terms of Payment"
        ],
      "Phone": ["Contact No", "Phone", "Phone No", "Contact","Mobile","Mob.","Mob","Phone :","Phone:","Cell No.","Cell No","Mobile No.","Mobile No"],
      "Reference":["Relerence","Reference"],
      "Total Amount": [
            "Grand Total",
            "Total Amount",
            "Total",
            "Total Payble Amount",
            "Net Amount",
            "Basic Amount",
            "Tax Amount"
        ],
        "Sub Total": ["Sub Total"],
        "CGST %": [
        "Central GST Rate",
        "CGST Percentage",
        "CGST (%)",
        "Central GST %",
        "C-GST Rate",
        "CGST %",
        "Central GST Percentage",
        "C.Gst %",
        "CGST(9.00%)"
      ],
        "S.Gst %": [
        "State GST Rate",
        "SGST Percentage",
        "SGST (%)",
        "State GST %",
        "S-GST Rate",
        "SGST %",
        "State GST Percentage",
        "S.Gst %",
        "SGST(9.00%)"
      ],
      "Qtn No":["Qtn. No."],
      "Qtn date":["Qtn. Date"],
      "Kind Attn.":["Kind Attn."],
      "M/S.":["M/S."],
      "Total Amount": [
        "Total Amount",
        "Grand Total",
        "Total",
        "Total Payble Amount",
        "Net Amount",
        "Basic Amount",
        "Tax Amount",
        "Grand Total INR",
        "Total Amount: INR",
        "Grand Total: INR",
        "Tax Amount(in words) INR"
      ],
      "Amount in words":["Amount In Words"],
      "Payment Terms": ["Payment Terms","Mode/Terms of Payment","Payment Terms:","Payment Terms :"], 
      "Delivery":["Delivery"],          # changes
      "Insurance":["Insurance"],
      "Delivery terms":["Delivery terms"],
      "Packaging":["PACKAGING","Packing"],
      "Test Report":["Test Report"],
      "PAN No": ["PAN No", "PAN Number","PAN NO","PAN NO:","PAN","PAN NO."]
    },
    "table_structure": {
        "columns": {
            "Sr no": [
                "SR.",
                "SR No.",
                "No.",
                "Sr.",
                "Sr No"
            ],
            "Item Description": [
                "Item Description",
                "Product Description",
                "Service Description",
                "Description",
                "Item Details",
                "Product Details",
                "Particulars",
                "Description of Goods",
                "Item Desc"
            ],
                  "HSN/SAC Code": [
        "HSN/SAC Code",
        "HSN No.",
        "HSN Code",
        "SAC Code",
        "Harmonized System of Nomenclature (HSN)",
        "Service Accounting Code (SAC)",
        "Tax Code",
        "Product Classification Code",
        "Service Classification Code",
        "HS Code",
        "Customs Code",
        "Item Tax Code",
        "Classification Code",
        "Goods/Service Code",
        "HSN/SAC Classification",
        "HSN/SAC Number",
        "HSN/SAC"
      ],
      "GST Rate %": [
        "GST Percentage",
        "GST Rate",
        "GST (%)",
        "GST % on Product",
        "GST Rate %",
        "GST",
        "GST %"
      ],
      "UOM": ["UOM","Unit of Measure","Measurement Unit","Unit","Unit Type", "Measurement Type","Unit Quantity","Product Unit","Service Unit","Quantity Unit","Measurement Unit Code","Product Measurement Unit","Service Measurement Unit","UOM Code"],
      "QTY.": [
        "QTY",
        "Quantity",
         "-Quantity"
        "Item Quantity",
        "Product Quantity",
        "No. of Items",
        "Total Quantity",
        "Ordered Quantity",
        "Quantity Ordered",
        "Quantity of Product",
        "Item Count",
        "Units Ordered",
        "Quantity of Items",
        "Product Units",
        "QTY."
      ],
      "Rate": [
        "Rate",
        "Unit Price",
        "Price per Unit",
        "Item Rate",
        "Product Rate",
        "Cost per Unit",
        "Basic Rate",
        "Rate (INR)"
      ],
      "Amount": ["Amount", "Total Amount", "Invoice Amount", "Net Amount", "Total Price", "Amount Due","Amount (INR)"]
        }
    }
 }
}

def preprocess_text(raw_text):
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', raw_text)
    # Normalize pipes and other delimiters
    cleaned_text = cleaned_text.replace('|', ' | ')
    # Strip unnecessary white spaces
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def convert_pdf_to_images(pdf_path, output_dir, dpi=300):
    os.makedirs(output_dir, exist_ok=True)
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f'page_{i + 1}.jpg')
        image.save(image_path, 'JPEG')
        image_paths.append(image_path)
    return image_paths

def extract_text_from_images(image_paths):
    extracted_text = ""
    for image_path in image_paths:
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            extracted_text += text
        except Exception as e:
            print(f"[ERROR] Failed to extract text from image {image_path}: {e}")
    # Debugging: Print extracted text before returning it
    print("[INFO] Extracted Text from Images:")
    print(extracted_text[:5000])  # Print first 500 characters

    return extracted_text

                                            # DATA PREPROCESSING / DATA CLEANSING

prohibited_date_fields = [
    'Vendor Name', 'Vendor Address', 'Contact', 'Freight', 'Transporter', 
    'Booking', 'Agent Code', 'Acknowledgement No.', 'Party P.O. Ref'
]

forbidden_keywords = ["Acknowledgement Date", "Acknowledgement", "GSTIN", "Numbe","SR. Item Code Item"]

allowed_date_fields = ["Acknowledgement Date", "Payment Terms"]

def remove_keys_from_value(value):
    all_keys = [re.escape(k) for keys in data_model["keys"].values() for k in keys]
    all_keys_pattern = r'\b(?:' + '|'.join(all_keys) + r')\b'
    
    cleaned_value = re.sub(all_keys_pattern, '', value, flags=re.IGNORECASE).strip()
    return cleaned_value

def remove_forbidden_keywords(value):
    keyword_pattern = r'\b(?:' + '|'.join(map(re.escape, forbidden_keywords)) + r')\b'
    return re.sub(keyword_pattern, '', value, flags=re.IGNORECASE).strip()
                                         
def clean_value(value, field_name):
    # General cleaning of the value, including removing unwanted characters
    value = re.sub(r'^[^\w\.:@/-]+|[^\w\.:@/-]+$', '', value)  # Remove non-alphanumeric symbols except important ones
    value = value.rstrip(" .")  # Trim trailing spaces and dots

    if field_name in prohibited_date_fields:
        value = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '', value).strip()  

    value = remove_forbidden_keywords(value)  # Remove any forbidden keywords
    return value.strip() 

def is_table_header(value):
    table_header_keywords = ["Item", "Code", "Description", "Quantity", "Price", "Amount"]      
    return any(keyword in value for keyword in table_header_keywords)

    #---------------------------- replace find_value_in_text---------------------------------------- 

def find_value_in_text(text, key_aliases, field_name, pdf_path):
    # Step 1: Try to find key coordinates
    key_coordinates = find_key_coordinates(pdf_path, key_aliases)
    
    # Step 2: Extract all keys pattern for further matching
    all_keys = [re.escape(k) for keys in data_model["keys"].values() for k in keys]
    all_keys_pattern = r'\b(?:' + '|'.join(all_keys) + r')\b'

    for alias in key_aliases:
        # Try to match the key and extract value
        pattern = re.escape(alias) + r'\s*[:\-]?\s*(.*?)(?=\s*\|)?(?=\n(?!.*' + all_keys_pattern + r')|(?=\n|$))'
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            value = match.group(1).strip()
            value = clean_value(value, field_name)
            value = remove_forbidden_keywords(value)

            # Clean value by removing any remaining key names
            for key in all_keys:
                if re.search(r'\b' + key + r'\b', value, re.IGNORECASE):
                    value = re.split(r'\b' + key + r'\b', value, maxsplit=1, flags=re.IGNORECASE)[0].strip()

            # Step 3: If coordinates found, return them, else return value only
            if key_coordinates:
                coordinates = get_value_coordinates_after_key(pdf_path, value, key_coordinates)
                return {"value": value, "coordinates": coordinates}
            else:
                return {"value": value, "coordinates": None}
    
    # Step 4: Fallback in case no match was found with primary pattern
    for alias in key_aliases:
        fallback_pattern = re.escape(alias) + r'\s*[:\-]?\s*(.*?)\s*$'
        fallback_match = re.search(fallback_pattern, text, re.IGNORECASE)

        if fallback_match:
            value = fallback_match.group(1).strip()
            value = clean_value(value, field_name)
            value = remove_forbidden_keywords(value)

            for key in all_keys:
                if re.search(r'\b' + key + r'\b', value, re.IGNORECASE):
                    value = re.split(r'\b' + key + r'\b', value, maxsplit=1, flags=re.IGNORECASE)[0].strip()

            # Step 5: Return value and coordinates if found, else just value
            if key_coordinates:
                coordinates = get_value_coordinates_after_key(pdf_path, value, key_coordinates)
                return {"value": value, "coordinates": coordinates}
            else:
                return {"value": value, "coordinates": None}
    
    # Step 6: If nothing found, return empty value
    return {"value": "", "coordinates": None}

def find_key_coordinates(pdf_path, key_aliases):
    pdf_document = fitz.open(pdf_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]

        for alias in key_aliases:
            key_instances = page.search_for(alias)
            if key_instances:
                for key_rect in key_instances:
                    coordinates = {
                        "page": page_num + 1,
                        "x": key_rect.x0,
                        "y": key_rect.y0,
                        "width": key_rect.width,
                        "height": key_rect.height
                    }
                    pdf_document.close()
                    return coordinates

    pdf_document.close()
    return None 

def get_value_coordinates_after_key(pdf_path, value, key_coordinates):
    pdf_document = fitz.open(pdf_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]

        if page_num + 1 == key_coordinates["page"]:
            
            search_area = fitz.Rect(
                key_coordinates["x"] + key_coordinates["width"],  
                key_coordinates["y"],  
                page.rect.width,       
                key_coordinates["y"] + 50  
            )
 
            # Search for the value within the defined search area
            value_instances = page.search_for(value, clip=search_area)

            if value_instances:
                # Get coordinates of the first matching instance
                for inst in value_instances:
                    rect = fitz.Rect(inst)
                    coordinates = {
                        "page": page_num + 1,
                        "x": rect.x0,
                        "y": rect.y0,
                        "width": rect.width,
                        "height": rect.height
                    }
                    pdf_document.close()
                    return coordinates  

    pdf_document.close()
    return {"page": None, "x": None, "y": None, "width": None, "height": None}

column_coordinates = {}

def get_coordinates_from_pdf(pdf_path, search_text, last_coords=None):
    pdf_document = fitz.open(pdf_path)
    coordinates = None

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]

        # If last coordinates are provided, limit the search to the area below those coordinates
        search_area = None
        if last_coords:
            search_area = fitz.Rect(last_coords["x"], last_coords["y"] + last_coords["height"], page.rect.width, page.rect.height)

        # Search for the exact text in the PDF
        text_instances = page.search_for(search_text, clip=search_area)

        # Check if any instances of the text were found
        if text_instances:
            # Extract the coordinates for the first matching instance
            for inst in text_instances:
                rect = fitz.Rect(inst)
                coordinates = {
                    "page": page_num + 1,
                    "x": rect.x0,
                    "y": rect.y0,
                    "width": rect.width,
                    "height": rect.height
                }
                break
      
        if coordinates:
            break

    pdf_document.close()
    return coordinates if coordinates else {"page": None, "x": None, "y": None, "width": None, "height": None}

def get_coordinates_for_numeric_fields(pdf_path, field_name, numeric_value, used_coordinates):
    pdf_document = fitz.open(pdf_path)
    coordinates = None  
    
    # Trim leading/trailing spaces from numeric_value
    numeric_value = numeric_value.strip()
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        
        numeric_instances = page.search_for(numeric_value)
        
        if not numeric_instances:
            numeric_instances = page.search_for(numeric_value.replace(",", ""))
        
        if not numeric_instances:
            cleaned_numeric_value = numeric_value.lstrip(".,")
            numeric_instances = page.search_for(cleaned_numeric_value)

        # **Avoid Coordinate Overlap**
        if numeric_instances:
            for num_inst in numeric_instances:
                num_rect = fitz.Rect(num_inst)
                new_coords = {
                    "page": page_num + 1,
                    "x": num_rect.x0,
                    "y": num_rect.y0,
                    "width": max(num_rect.width, 10),  # Ensure minimum width (Threshold 10)
                    "height": num_rect.height
                }

                # **Ensure coordinates are unique**
                if not any(
                    abs(new_coords["x"] - uc["x"]) < 5 and abs(new_coords["y"] - uc["y"]) < 5
                    for uc in used_coordinates
                ):
                    coordinates = new_coords
                    used_coordinates.append(new_coords)  # Track used coordinates
                    break  

            if coordinates:
                break  # Ek baar value mil gayi toh aage pages check karne ki zaroorat nahi

    pdf_document.close()
    
    # **Return None if coordinates not found**
    return coordinates if coordinates else {"page": None, "x": None, "y": None, "width": None, "height": None}


def get_row_coordinates_from_pdf(pdf_path, row_data):
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

    for field, field_data in row_data.items():
        if isinstance(field_data, dict) and field_data.get("coordinates"):
            coords = field_data.get("coordinates")
            if coords and coords["x"] is not None and coords["y"] is not None:
                min_x = min(min_x, coords["x"])
                min_y = min(min_y, coords["y"])
                max_x = max(max_x, coords["x"] + coords["width"])
                max_y = max(max_y, coords["y"] + coords["height"])

    if min_x < float('inf') and min_y < float('inf'):
        return {
            "page": row_data.get("sr_no", {}).get("coordinates", {}).get("page",1),
            "x": min_x,
            "y": min_y,
            "width": max_x - min_x,
            "height": max_y - min_y
        }
    return None

def combine_row_coordinates(row_coords_list):
    combined_x = float('inf')
    combined_y = float('inf')
    combined_right = -float('inf')
    combined_bottom = -float('inf')

    # row_coords_list is assumed to be a list of coordinate dictionaries,
    # each with keys "x", "y", "width", and "height"
    for coords in row_coords_list:
        if (coords.get("x") is not None and coords.get("y") is not None and
            coords.get("width") is not None and coords.get("height") is not None):
            combined_x = min(combined_x, coords["x"])
            combined_y = min(combined_y, coords["y"])
            combined_right = max(combined_right, coords["x"] + coords["width"])
            combined_bottom = max(combined_bottom, coords["y"] + coords["height"])

    if combined_x == float('inf') or combined_y == float('inf'):
        return None

    return {
        "page": row_coords_list[0].get("page", 1), # Assumes all rows are on the same page
        "x": combined_x,
        "y": combined_y,
        "width": combined_right - combined_x,
        "height": combined_bottom - combined_y
    }

def is_text_based(pdf_path):
    """Check if the PDF contains selectable text using PyMuPDF (fitz)."""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text = page.get_text("text")
            if text and text.strip():
                return True
    except Exception as e:
        print(f"[ERROR] Failed to check if PDF is text-based: {e}")
        
    return False

def preprocess_extracted_text(text):
    """Remove extra spaces, unwanted line breaks, and unstructured content."""
    cleaned_text = re.sub(r'\n+', ' ', text)
    cleaned_text = re.sub(r'(\s{2,})', ' ', cleaned_text)
    return cleaned_text.strip()

def extract_table_from_image(pdf_path):
    """Extract tables from an image-based PDF using OCR."""
    images = convert_from_path(pdf_path)  # Convert PDF to images
    extracted_tables = []

    for i, image in enumerate(images):
        img = np.array(image)
        binary = preprocess_image(img)  # Apply preprocessing

        custom_config = "--psm 6"
        text = pytesseract.image_to_string(binary, config=custom_config)

        text = preprocess_extracted_text(text)

        lines = text.split("\n")
        detected_table = [line.strip().split() for line in lines if line.strip()]

        if detected_table:
            extracted_tables.append(detected_table)

    return extracted_tables

def clean_text(text):
    """Normalize text by removing extra spaces and special characters."""
    return re.sub(r"\s+", " ", text).strip().lower()

def match_table_headers(extracted_tables, template_structures):
    """Find the best-matching table structure based on headers."""
    best_match_table_structure = None
    highest_match_score = 0

    for template_structure in template_structures:
        if not template_structure or 'columns' not in template_structure:
            # print("[ERROR] Missing 'columns' in table structure:", template_structure)
            continue  # Skip this template if 'columns' is missing

        template_headers = [clean_text(col) for col in template_structure['columns']]
        
        for table in extracted_tables:
            if not table:
                continue

            first_row = table[0]  # Assume first row contains headers
            extracted_headers = [clean_text(cell) for cell in first_row]

            # Apply a higher threshold for matching headers (e.g., 75 or 80)
            match_score = fuzz.token_sort_ratio(" ".join(extracted_headers), " ".join(template_headers))
            
            # print(f"[DEBUG] Match Score: {match_score} | Extracted Headers: {extracted_headers} | Template Headers: {template_headers}")

            if match_score > 70:  # Adjust the threshold as needed
                if match_score > highest_match_score:
                    highest_match_score = match_score
                    best_match_table_structure = template_structure

    return best_match_table_structure

def extract_table_data(pdf_path, table_structure):
    """Extract table data based on the provided table structure."""
    table_data = []
    seen_sr_numbers = set()
    last_coordinates = {}
    used_coordinates = []

    try:
        # Use Camelot to extract tables from PDF
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")

        if tables.n == 0:
            print("[INFO] No tables detected with lattice. Trying 'stream' mode...")
            tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")

        print(f"[DEBUG] Total tables detected: {tables.n}")

        if tables.n == 0:
            print("[WARNING] No tables found in the PDF.")
            return table_data, None

        pdf_document = fitz.open(pdf_path)

        for table in tables:
            extracted_table = table.df.values.tolist()
            if not extracted_table:
                continue

            print(f"[DEBUG] Extracted Table Content: {extracted_table}")

            # Assuming headers are present in the first row, you can map them based on table_structure
            header_row = table_structure.get('columns', [])
            header_index = 0  # Default to first row as header

            valid_headers = set(header_row)

            for row in extracted_table[header_index + 1:]:
                if not any(cell.strip() for cell in row):  # Skip empty rows
                    continue

                split_row = [cell.split('\n') if '\n' in cell else [cell] for cell in row]
                max_split_count = max(len(cells) for cells in split_row)

                # Pad shorter columns to match the max_split_count
                for i in range(len(split_row)):
                    split_row[i] += [''] * (max_split_count - len(split_row[i]))

                for row_idx in range(max_split_count):
                    row_dict = {}

                    for idx in range(len(header_row)):
                        field_name = header_row[idx]
                        field_value = split_row[idx][row_idx]

                        if field_name.lower().replace(" ", "").replace(".", "") in ["srno", "sr"]:
                            field_coordinates = {"page": None, "x": None, "y": None, "width": None, "height": None}
                        else:
                            if field_name not in last_coordinates:
                                field_coordinates = get_coordinates_from_pdf(pdf_path, field_value)
                            else:
                                field_coordinates = get_coordinates_for_numeric_fields(
                                    pdf_path, field_name, field_value, used_coordinates
                                )

                            # Update last_coordinates for the column
                            if field_coordinates and field_coordinates["x"] is not None:
                                last_coordinates[field_name] = field_coordinates
                                used_coordinates.append(field_coordinates)

                        if field_name and field_value:
                            row_dict[field_name] = {
                                "value": field_value,
                                "coordinates": field_coordinates
                            }

                    row_coordinates = get_row_coordinates_from_pdf(pdf_path, row_dict)
                    if row_coordinates:
                        row_dict["row_coordinates"] = row_coordinates

                    # Check for serial number and ensure it's unique
                    sr_no_value = ""
                    for key in row_dict:
                        key_norm = key.lower().replace(" ", "").replace(".", "")
                        if key_norm in ["srno", "sr"]:
                            sr_no_value = row_dict[key]["value"].strip()
                            break

                    if sr_no_value and sr_no_value.isdigit() and sr_no_value not in seen_sr_numbers:
                        table_data.append(row_dict)
                        seen_sr_numbers.add(sr_no_value)

        pdf_document.close()

    except Exception as e:
        print(f"[ERROR] Failed to extract tables: {e}")

    all_row_coords = [row["row_coordinates"] for row in table_data if "row_coordinates" in row]
    combined_row_coordinates = combine_row_coordinates(all_row_coords) if all_row_coords else None
    print(f"[INFO] Combined Row Coordinates: {combined_row_coordinates}")

    return table_data, combined_row_coordinates

def extract_invoice_data_with_table(pdf_path, text, template_structures):
    """Extract invoice data and specific table based on PDF type."""
    if not text:
        return {}

    # Step 1: Match the best table template
    extracted_tables = extract_table_data(pdf_path,table_structure)  # Assuming this method extracts tables
    best_match_template = match_table_headers(extracted_tables, template_structures)

    if not best_match_template:
        print("[ERROR] No matching table structure found.")
        return {}

    # print("[INFO] Matched Template Structure:", best_match_template)

    extracted_data = {}

    # Step 2: Extract data using the matched template
    # Extract other data based on the template (keys and aliases)
    for key, aliases in best_match_template['keys'].items():
        result = find_value_in_text(text, aliases, key, pdf_path)
        value = result['value']
        extracted_data[key] = value or ""
        extracted_data[f"{key}_coordinates"] = result['coordinates'] if value else {
            "page": None, "x": None, "y": None, "width": None, "height": None
        }

    # Step 3: Extract table data using the 'table_structure' from the matched template
    table_structure = best_match_template['table_structure']
    table_data, combined_row_coordinates = extract_table_data(pdf_path, table_structure)  # New method

    # Step 4: Store table data and coordinates in the result
    extracted_data['Table Data'] = table_data
    extracted_data['Combined Row Coordinates'] = combined_row_coordinates

    return extracted_data
        
#---------------------------------Changes in validate_extracted_data , extract_matching_substring , find_nearest_match_in_box_using_regex , calculate_proximity, calculate_template_score-------------------------------------------

def extract_matching_substring(value, pattern):            
    """
    Extract the first matching substring from the value using the given regex pattern.
    If no match is found, return None.
    """
    matches = list(re.finditer(pattern, value))
    if matches:
        # Take the first match found
        matched_value = matches[0].group(0)
        # print(f"[INFO] Found matching substring: '{matched_value}' in value: '{value}'")
        return matched_value
    # print(f"[INFO] No matching substring found in value: '{value}' for pattern: {pattern}")
    return None

def validate_extracted_data(template_result, extracted_data, pdf_data, regex):
    """
    Validate and clean the extracted data using regex patterns. If no valid match is found,
    clear the value and mark it as invalid.
    """
    if not isinstance(regex, dict):
        raise ValueError(f"Expected 'regex' to be a dictionary, but got {type(regex).__name__}")

    updates = {}  # Collect updates to apply to the extracted data

    for key, value in extracted_data.items():
        if "_coordinates" in key or key in {"Table Data", "Combined Row Coordinates"}:
            # print(f"[INFO] Skipping key: {key} (coordinate or non-validatable field)")
            continue

        pattern = regex.get(key)
        if not pattern:
            print(f"[WARNING] No regex pattern for key: {key}")
            continue

        coordinates_key = f"{key}_coordinates"
        coordinates = extracted_data.get(coordinates_key, {})
        value = str(value).strip()

        print(f"Validating key: {key}, value: '{value}' with pattern: {pattern}")

        # Full match validation
        match = re.fullmatch(pattern, value)
        if match:
            matched_value = match.group(0)
            extracted_data[key] = matched_value  # Save the fully matched value
            updates[f"{key}_is_valid"] = True
            print(f"[INFO] Validated key: {key}, value: '{matched_value}' (full match)")
            continue

        # Substring extraction if no full match is found
        print(f"[INFO] No full match found, checking for valid substrings for key: {key}")
        matched_value = extract_matching_substring(value, pattern)
        if matched_value:
            extracted_data[key] = matched_value
            updates[f"{key}_is_valid"] = True
            print(f"[INFO] Extracted matching substring for key: {key}, value: '{matched_value}'")
            continue

        # Nearby text search if no substring match is found
        print(f"[INFO] No valid substring found in value: '{value}', searching nearby text for key: {key}")
        matched_value = find_nearest_match_in_box_using_regex(value, extracted_data, pdf_data, pattern, coordinates)
        if matched_value and matched_value['value']:
            extracted_data[key] = matched_value['value']
            updates[f"{key}_is_valid"] = True
            print(f"[INFO] Found valid match for key: {key} in nearby text: {matched_value['value']}")
        else:
            # Clear the value and flag as invalid if no match is found
            extracted_data[key] = ""  # Clear the invalid value
            updates[f"{key}_is_valid"] = False
            print(f"[INFO] No valid match found for key: {key}. Value cleared and flagged as invalid.")

    # Apply all collected updates after iteration
    extracted_data.update(updates)
    return template_result

def find_nearest_match_in_box_using_regex(value, extracted_data, pdf_data, pattern, key_coordinates, margin_x=50, margin_y=50):
   
    if not key_coordinates or len(key_coordinates) != 2:
        print(f"[ERROR] Invalid key_coordinates: {key_coordinates}")
        return {"value": "", "coordinates": None}

    x, y = key_coordinates
    box = {
        "x_min": x - margin_x,
        "x_max": x + margin_x,
        "y_min": y - margin_y,
        "y_max": y + margin_y,
    }

    print(f"[INFO] Searching for matches near bounding box: {box}")

    closest_match = None
    min_proximity = float('inf')

    for line_data in pdf_data:
        text = line_data.get("text", "").strip()
        line_coordinates = line_data.get("coordinates")

        if not line_coordinates or len(line_coordinates) != 2:
            continue

        x_line, y_line = line_coordinates
        if box["x_min"] <= x_line <= box["x_max"] and box["y_min"] <= y_line <= box["y_max"]:
            match = re.search(pattern, text)  # Match directly against the text
            if match:
                proximity = calculate_proximity(key_coordinates, line_coordinates)
                if proximity < min_proximity:
                    closest_match = {"value": match.group(0), "coordinates": line_coordinates}
                    min_proximity = proximity

    if closest_match:
        print(f"[INFO] Closest match found: '{closest_match['value']}' at {closest_match['coordinates']}")
        return closest_match
    else:
        print("[INFO] No valid match found within bounding box.")
        return {"value": "", "coordinates": key_coordinates}

def calculate_proximity(coord1, coord2):
    
    if not coord1 or not coord2:
        return float('inf') 

    x1, y1 = coord1
    x2, y2 = coord2
    proximity = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    print(f"[INFO] Proximity between {coord1} and {coord2}: {proximity}")
    return proximity

def calculate_template_score(extracted_data, template):
   
    keys = template.get("keys", {})
    matched_keys = 0
    total_keys = len(keys)

    # Check for key matches and aliases
    for key, aliases in keys.items():
        if extracted_data.get(key):  # Direct match
            matched_keys += 1
        elif any(extracted_data.get(alias) for alias in aliases):  # Alias match
            matched_keys += 0.8  # Partial score for alias match

    key_score = (matched_keys / total_keys) if total_keys else 0
    print(f"[INFO] Key matching score: {key_score:.3f} (matched: {matched_keys:.1f} / total: {total_keys})")

    # Table matching logic
    table_columns = template.get("table_structure", {}).get("columns", [])
    table_data = extracted_data.get("table", [])
    matched_columns = 0
    total_columns = len(table_columns)

    if table_data and total_columns > 0:    
        for column in table_columns:
            if any(row.get(column) for row in table_data):
                matched_columns += 1

    table_score = (matched_columns / total_columns) if total_columns else 0
    print(f"[INFO] Table column matching score: {table_score:.3f} (matched: {matched_columns} / total: {total_columns})")

    # Adjust weights for key and table matching
    final_score = (key_score * 0.7) + (table_score * 0.3)  # Give higher weight to key matching
    print(f"[INFO] Final template score: {final_score:.3f}")
    return round(final_score, 3)

#-------------------------------------- all new below , replace with your old code complete----------------------------------------------------------------------- 

def create_new_template(existing_template, new_coordinates):
    
    new_template = existing_template.copy()  # Copy the existing template to keep the structure
    for key, value in new_template.items():
        if "coordinates" in value:  # If the key contains coordinates
            value["coordinates"] = new_coordinates.get(key, value["coordinates"])
    return new_template

def fetch_existing_template(template_id, templates):
    
    return templates.get(template_id)

def fetch_new_payload(api_url):
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()  # Parse the API response as JSON
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch data from API: {e}")
        return None
    except ValueError as e:
        print(f"[ERROR] Invalid JSON response: {e}")
        return None

def are_coordinates_different(existing_template, new_coordinates):
    
    for key, value in existing_template.items():
        if "coordinates" in value:
            existing_coords = value["coordinates"]
            new_coords = new_coordinates.get(key)
            if existing_coords != new_coords:
                return True
    return False

def get_next_template_id(templates):
    
    existing_ids = [int(t.split("_")[1]) for t in templates.keys() if t.startswith("template_")]
    next_id = max(existing_ids, default=0) + 1  # Get the next template number
    return f"template_{next_id}"

def generate_new_template_from_payload(api_url, templates):
    
    # Fetch the new payload from the API
    new_payload = fetch_new_payload(api_url)
    if not new_payload:
        print("[ERROR] Failed to fetch new payload.")
        return None

    # Extract the template ID dynamically from the payload
    template_id = new_payload.get("template_id")
    if not template_id:
        print("[ERROR] Template ID not found in the payload.")
        return None

    print(f"[INFO] Template ID identified: {template_id}")

    # Fetch the existing template by ID
    existing_template = fetch_existing_template(template_id, templates)
    if not existing_template:
        print(f"[ERROR] Template {template_id} not found in the database.")
        return None

    # Extract new coordinates from the payload
    new_coordinates = {key: value.get("coordinates") for key, value in new_payload.get("fields", {}).items()}

    # Check if coordinates are different
    if not are_coordinates_different(existing_template, new_coordinates):
        print("[INFO] Coordinates are the same. No new template needed.")
        return None

    # Create a new template by updating the coordinates of the existing template
    new_template = create_new_template(existing_template, new_coordinates)

    # Assign a new template ID
    new_template_id = get_next_template_id(templates)

    # Save the new template in the database
    templates[new_template_id] = new_template

    print(f"[INFO] New template created successfully with ID: {new_template_id}")
    return {new_template_id: new_template}

# Replace with the actual API URL
api_url = "https://example.com/api/new-payload"        # will be replaced by actual api here

# Generate the new template dynamically
new_template = generate_new_template_from_payload(api_url, templates)

# Output the result
if new_template:
    print(json.dumps(new_template, indent=6))
else:
    print("No new template created.")

def process_invoice_with_all_templates(pdf_path, output_dir, templates, regex):
    all_template_results = []

    image_paths = convert_pdf_to_images(pdf_path, output_dir)  # Convert PDF to images
    pdf_data = extract_text_from_images(image_paths)  # Extract text from the images

    # Debugging: Check the extracted text
    if not pdf_data:
        print("[ERROR] No text extracted from the images.")
        return

    for template_key, template in templates.items():
        print(f"[INFO] Processing template: {template_key}")

        # Debugging: Check what text is being passed to the function
        # print(f"[INFO] Extracted text: {pdf_data[:500]}")  

        # Make sure the template is passed as an argument, and also pass the text
        extracted_data = extract_invoice_data_with_table(pdf_path, pdf_data, template) 

        score = calculate_template_score(extracted_data, template)
        all_template_results.append({
            "template_id": template_key,
            "score": score,
            "extracted_data": extracted_data
        })

    # Sort templates by score in descending order
    sorted_results = sorted(all_template_results, key=lambda x: x['score'], reverse=True)

    if sorted_results:
        best_template_result = sorted_results[0]
        print(f"[INFO] Best template: {best_template_result['template_id']} with score: {best_template_result['score']}")
        
        # Validate the extracted data
        validated_result = validate_extracted_data(best_template_result, best_template_result['extracted_data'], pdf_data, regex)
        return validated_result

    return None

if __name__ == "__main__":        
    pdf_path = r'/Users/quantumitservicesllp/Desktop/invoice_project/invoices/invoice_6.pdf'         
    output_dir = r'/Users/quantumitservicesllp/Desktop/invoice_project/invoices/images'
    regex = data_model.get("regex",{})

    api_url = "---"  # Replace with API URL 

    try:
        # Fetch new payload from the API
        new_payload = fetch_new_payload(api_url)

        if new_payload:
            # Fetch the template ID from the payload 
            template_id = new_payload.get("template_id")
            if template_id:
                existing_template = fetch_existing_template(template_id, templates)

                if existing_template:
                    print(f"Found existing template: {template_id}")
                    # Generate a new template with updated coordinates
                    new_coordinates = {key: value.get("coordinates") for key, value in new_payload.get("fields", {}).items()}
                    updated_template = create_new_template(existing_template, new_coordinates)

                    print("New template created successfully.")

                    # Process the invoice with the updated template
                    validated_result = process_invoice_with_all_templates(pdf_path, output_dir, {template_id: updated_template}, regex)
     
                    # Save the validated result
                    if validated_result:
                        with open('all_template_results.json', 'w') as result_file:
                            json.dump(validated_result, result_file, indent=6)
                        print("Validated result saved to all_template_results.json")
                else:
                    print(f"[ERROR] Template {template_id} not found.")
        
            else:
                print("[ERROR] Template ID not found in the payload.")
        else:
            print("No payload found. Processing with current templates.")
            validated_result = process_invoice_with_all_templates(pdf_path, output_dir, templates, regex)

            # Save the validated result
            if validated_result:
                with open('all_template_results.json', 'w') as result_file:
                    json.dump(validated_result, result_file, indent=6)
                print("[INFO] Validated result saved to all_template_results.json")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")

