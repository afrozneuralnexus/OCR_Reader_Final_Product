import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
import pytesseract
from pytesseract import Output
from skimage.filters import threshold_local
from PIL import Image
import pandas as pd
from datetime import datetime
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import io
import tempfile
import shutil

# Configure Tesseract path for different environments
def setup_tesseract():
    """Configure Tesseract OCR path based on the environment"""
    # Check if tesseract is in PATH
    if shutil.which('tesseract'):
        return True
    
    # Common Tesseract installation paths
    possible_paths = [
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract',
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        '/opt/homebrew/bin/tesseract',  # macOS ARM
        '/usr/local/Cellar/tesseract/*/bin/tesseract',  # macOS Intel
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return True
    
    return False

# Setup Tesseract on app start
if not setup_tesseract():
    st.error("⚠️ Tesseract OCR is not installed. Please check the installation guide.")

# Constants for file names
MASTER_EXCEL_FILE = "invoices_database.xlsx"
MODEL_FILE = "invoice_processor_model.pkl"

def load_and_preprocess_image(image_file):
    """Load and preprocess the image with enhanced preprocessing"""
    # Convert uploaded file to numpy array
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError(f"Could not load image")

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image if too large for better processing
    height, width = image.shape[:2]
    if height > 2000 or width > 2000:
        scale = min(2000/height, 2000/width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return image

def enhance_image_quality(image):
    """Enhanced image quality improvement for better OCR"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Multiple enhancement techniques
    # 1. Denoising
    denoised = cv2.medianBlur(gray, 3)

    # 2. Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(denoised)

    # 3. Adaptive thresholding
    T = threshold_local(contrast_enhanced, 15, offset=12, method="gaussian")
    binary = (contrast_enhanced > T).astype("uint8") * 255

    # 4. Morphological operations to clean up the image
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary, contrast_enhanced, gray

def extract_text_from_image(image):
    """Enhanced text extraction with multiple OCR configurations"""
    configs = [
        r'--oem 3 --psm 6',
        r'--oem 3 --psm 4',
        r'--oem 3 --psm 3',
    ]

    best_text = ""
    best_config = ""

    for config in configs:
        try:
            text = pytesseract.image_to_string(image, config=config)
            if len(text) > len(best_text):
                best_text = text
                best_config = config
        except Exception as e:
            if "tesseract is not installed" in str(e).lower():
                raise RuntimeError("Tesseract OCR is not properly installed. Please ensure Tesseract is installed on your system.")
            continue

    if not best_text:
        raise RuntimeError("Could not extract text from image. Please ensure the image is clear and readable.")

    try:
        detailed_data = pytesseract.image_to_data(image, output_type=Output.DICT, config=best_config)
    except:
        detailed_data = {}

    return best_text, detailed_data

def extract_invoice_number(text):
    """Extract invoice number"""
    patterns = [
        r'Invoice\s*no\s*:?\s*(\d+)',
        r'Invoice\s*#?\s*:?\s*(\d+)',
        r'Invoice\s*Number\s*:?\s*(\d+)',
        r'INV\s*:?\s*(\d+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0]

    return ""

def extract_date_of_issue(text):
    """Extract date of issue"""
    patterns = [
        r'Date\s*of\s*issue\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        r'Issue\s*Date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        r'Date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        r'Date\s+of\s+issue\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                match_start = text.find(match)
                if match_start != -1:
                    context_start = max(0, match_start - 50)
                    context_end = min(len(text), match_start + len(match) + 50)
                    context_text = text[context_start:context_end]
                    if re.search(r'(date|issue)', context_text, re.IGNORECASE):
                        return match
            return matches[0]

    lines = text.split('\n')
    for i, line in enumerate(lines):
        if 'date of issue' in line.lower() or 'issue date' in line.lower():
            search_text = line
            if i + 1 < len(lines):
                search_text += " " + lines[i + 1]
            date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})', search_text)
            if date_match:
                return date_match.group(1)

    return ""

def extract_seller_info(text):
    """Extract seller information"""
    seller_name = ""
    seller_address = ""

    seller_match = re.search(r'Seller\s*:?\s*\n\s*([^\n]+)', text, re.IGNORECASE)
    if seller_match:
        seller_name = seller_match.group(1).strip()

        lines = text.split('\n')
        seller_idx = -1
        for i, line in enumerate(lines):
            if seller_name in line:
                seller_idx = i
                break

        if seller_idx >= 0 and seller_idx + 1 < len(lines):
            address_lines = []
            for i in range(seller_idx + 1, min(seller_idx + 4, len(lines))):
                line = lines[i].strip()
                if line and not line.lower().startswith('client') and not line.lower().startswith('tax'):
                    address_lines.append(line)
                else:
                    break
            seller_address = ', '.join(address_lines)

    return f"{seller_name}, {seller_address}".strip(', ')

def extract_client_info(text):
    """Extract client information"""
    client_name = ""
    client_address = ""

    client_match = re.search(r'Client\s*:?\s*\n\s*([^\n]+)', text, re.IGNORECASE)
    if client_match:
        client_name = client_match.group(1).strip()

        lines = text.split('\n')
        client_idx = -1
        for i, line in enumerate(lines):
            if client_name in line:
                client_idx = i
                break

        if client_idx >= 0 and client_idx + 1 < len(lines):
            address_lines = []
            for i in range(client_idx + 1, min(client_idx + 3, len(lines))):
                line = lines[i].strip()
                if line and not line.lower().startswith('tax') and not line.lower().startswith('items'):
                    address_lines.append(line)
                else:
                    break
            client_address = ', '.join(address_lines)

    return f"{client_name}, {client_address}".strip(', ')

def extract_summary_totals(text):
    """Extract VAT%, Net Worth, VAT, and Gross Worth from SUMMARY section"""
    vat_percent = ""
    net_worth = ""
    vat_amount = ""
    gross_worth = ""

    lines = text.split('\n')

    for i, line in enumerate(lines):
        if 'vat' in line.lower():
            percent_match_line = re.search(r'(\d+)\s*%', line)
            if percent_match_line:
                vat_percent = percent_match_line.group(1) + "%"
                break
            for j in range(i + 1, min(i + 4, len(lines))):
                percent_match_next = re.search(r'(\d+)\s*%', lines[j])
                if percent_match_next:
                    vat_percent = percent_match_next.group(1) + "%"
                    break
            if vat_percent:
                break

    for line in lines:
        if 'total' in line.lower():
            match = re.search(r'total.*?(?:\$?\s*([\d\s,]+\.?\d+))\s+(?:\$?\s*([\d\s,]+\.?\d+))\s+(?:\$?\s*([\d\s,]+\.?\d+))', line, re.IGNORECASE)
            if match:
                net_worth = "$" + match.group(1).replace(" ", "").strip()
                vat_amount = "$" + match.group(2).replace(" ", "").strip()
                gross_worth = "$" + match.group(3).replace(" ", "").strip()
                break

    if not (net_worth and vat_amount and gross_worth):
        text_lower = text.lower()

        net_match = re.search(r'(?:net\s*worth|amount\s*due)\s*[:\s]*\$?([\d\s,]+\.?\d+)', text_lower)
        if net_match:
            net_worth = "$" + net_match.group(1).replace(" ", "").strip()

        vat_amount_match = re.search(r'\bvat\b(?!\s*[%])\s*[:\s]*\$?([\d\s,]+\.?\d+)', text_lower)
        if vat_amount_match:
            vat_amount = "$" + vat_amount_match.group(1).replace(" ", "").strip()

        gross_match = re.search(r'(?:gross\s*worth|total\s*amount)\s*[:\s]*\$?([\d\s,]+\.?\d+)', text_lower)
        if gross_match:
            gross_worth = "$" + gross_match.group(1).replace(" ", "").strip()

    return vat_percent, net_worth, vat_amount, gross_worth

class InvoiceProcessor:
    """Model class to handle invoice processing and saving"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.processed_data = []
        self.model_version = "1.0"
        self.is_fitted = False

    def load_existing_model(self):
        """Load existing model if available"""
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, 'rb') as f:
                    model_data = pickle.load(f)
                self.vectorizer = model_data['vectorizer']
                self.processed_data = model_data['processed_data']
                self.model_version = model_data['model_version']
                self.is_fitted = model_data.get('is_fitted', False)
                return True
            except Exception as e:
                st.warning(f"Could not load existing model: {str(e)}")
                return False
        return False

    def fit(self, texts):
        """Fit the vectorizer on text data"""
        if texts:
            try:
                self.vectorizer.fit(texts)
                self.is_fitted = True
            except Exception as e:
                st.warning(f"Error fitting vectorizer: {str(e)}")

    def add_data(self, invoice_no, raw_text, image_name):
        """Add new data to the processor"""
        self.processed_data.append({
            'invoice_no': invoice_no,
            'raw_text': raw_text,
            'image_path': image_name,
            'timestamp': datetime.now()
        })

    def save_model(self, filename=MODEL_FILE):
        """Save the trained model"""
        model_data = {
            'vectorizer': self.vectorizer,
            'processed_data': self.processed_data,
            'model_version': self.model_version,
            'is_fitted': self.is_fitted,
            'timestamp': datetime.now()
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

        return filename

def process_invoice(image_file, processor=None):
    """Main function to process invoice and extract structured data"""
    original_image = load_and_preprocess_image(image_file)

    binary_image, contrast_enhanced, gray_image = enhance_image_quality(original_image)

    text_binary, _ = extract_text_from_image(binary_image)
    text_contrast, _ = extract_text_from_image(contrast_enhanced)
    text_gray, _ = extract_text_from_image(gray_image)

    texts = [text_binary, text_contrast, text_gray]
    final_text = max(texts, key=len)

    invoice_no = extract_invoice_number(final_text)
    date_of_issue = extract_date_of_issue(final_text)
    seller = extract_seller_info(final_text)
    client = extract_client_info(final_text)
    vat_percent, net_worth, vat_amount, gross_worth = extract_summary_totals(final_text)

    if processor:
        processor.add_data(invoice_no, final_text, image_file.name)

    return {
        'invoice_no': invoice_no,
        'date_of_issue': date_of_issue,
        'seller': seller,
        'client': client,
        'vat_percent': vat_percent,
        'net_worth': net_worth,
        'vat': vat_amount,
        'gross_worth': gross_worth,
        'raw_text': final_text,
        'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'image_filename': image_file.name,
        'images': {
            'original': original_image,
            'gray': gray_image,
            'contrast': contrast_enhanced,
            'binary': binary_image
        }
    }

def append_to_master_excel(result):
    """Append new invoice data to the master Excel file"""
    record = {
        'Invoice No': result['invoice_no'],
        'Date of Issue': result['date_of_issue'],
        'Seller': result['seller'],
        'Client': result['client'],
        'VAT (%)': result['vat_percent'],
        'Net Worth': result['net_worth'],
        'VAT': result['vat'],
        'Gross Worth': result['gross_worth'],
        'Processing Date': result['processing_date'],
        'Image Filename': result['image_filename']
    }

    if os.path.exists(MASTER_EXCEL_FILE):
        try:
            existing_df = pd.read_excel(MASTER_EXCEL_FILE)
        except Exception as e:
            existing_df = pd.DataFrame()

        new_df = pd.DataFrame([record])
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = pd.DataFrame([record])

    updated_df.to_excel(MASTER_EXCEL_FILE, index=False)

    return MASTER_EXCEL_FILE, len(updated_df)

# Streamlit UI
def main():
    st.set_page_config(page_title="Invoice Processing System", page_icon="📄", layout="wide")
    
    st.title("📄 Invoice Processing System")
    st.markdown("---")
    
    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = InvoiceProcessor()
        st.session_state.processor.load_existing_model()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Info")
        
        # Check Tesseract status
        try:
            tesseract_version = pytesseract.get_tesseract_version()
            st.success(f"✅ Tesseract {tesseract_version}")
        except Exception as e:
            st.error("❌ Tesseract not found")
            st.info("Please ensure Tesseract OCR is installed on your system.")
        
        if os.path.exists(MASTER_EXCEL_FILE):
            df = pd.read_excel(MASTER_EXCEL_FILE)
            st.metric("Total Invoices", len(df))
        else:
            st.metric("Total Invoices", 0)
        
        st.metric("Model Records", len(st.session_state.processor.processed_data))
        
        st.markdown("---")
        st.header("⬇️ Download Files")
        
        if os.path.exists(MASTER_EXCEL_FILE):
            with open(MASTER_EXCEL_FILE, 'rb') as f:
                st.download_button(
                    label="📥 Download Database",
                    data=f,
                    file_name=MASTER_EXCEL_FILE,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, 'rb') as f:
                st.download_button(
                    label="🤖 Download Model",
                    data=f,
                    file_name=MODEL_FILE,
                    mime="application/octet-stream"
                )
    
    # Main content
    tab1, tab2 = st.tabs(["📤 Upload & Process", "📊 View Database"])
    
    with tab1:
        st.header("Upload Invoice Images")
        uploaded_files = st.file_uploader(
            "Choose invoice images...", 
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("🔄 Process Invoices", type="primary"):
                all_results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing: {uploaded_file.name}")
                    
                    try:
                        # Process invoice
                        results = process_invoice(uploaded_file, st.session_state.processor)
                        all_results.append(results)
                        
                        # Append to database
                        excel_file, total_count = append_to_master_excel(results)
                        
                        # Display results
                        with st.expander(f"✅ {uploaded_file.name} - Invoice #{results['invoice_no']}", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("📋 Extracted Data")
                                st.write(f"**Invoice No:** {results['invoice_no'] or 'Not found'}")
                                st.write(f"**Date of Issue:** {results['date_of_issue'] or 'Not found'}")
                                st.write(f"**Seller:** {results['seller'] or 'Not found'}")
                                st.write(f"**Client:** {results['client'] or 'Not found'}")
                                st.write(f"**VAT (%):** {results['vat_percent'] or 'Not found'}")
                                st.write(f"**Net Worth:** {results['net_worth'] or 'Not found'}")
                                st.write(f"**VAT:** {results['vat'] or 'Not found'}")
                                st.write(f"**Gross Worth:** {results['gross_worth'] or 'Not found'}")
                            
                            with col2:
                                st.subheader("🖼️ Image Preview")
                                st.image(results['images']['original'], caption="Original Image", use_container_width=True)
                            
                            # Show processing steps
                            st.subheader("🔍 Processing Steps")
                            img_cols = st.columns(4)
                            with img_cols[0]:
                                st.image(results['images']['original'], caption="Original", use_container_width=True)
                            with img_cols[1]:
                                st.image(results['images']['gray'], caption="Grayscale", use_container_width=True)
                            with img_cols[2]:
                                st.image(results['images']['contrast'], caption="Enhanced", use_container_width=True)
                            with img_cols[3]:
                                st.image(results['images']['binary'], caption="Binary", use_container_width=True)
                            
                            # Show extracted text
                            with st.expander("📄 View Extracted Text"):
                                st.text(results['raw_text'])
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("✅ Processing complete!")
                
                # Update model
                if all_results:
                    all_texts = [result['raw_text'] for result in all_results]
                    
                    if st.session_state.processor.processed_data:
                        existing_texts = [data['raw_text'] for data in st.session_state.processor.processed_data]
                        all_texts = existing_texts + all_texts
                    
                    st.session_state.processor.fit(all_texts)
                    st.session_state.processor.save_model()
                    
                    st.success(f"✅ Processed {len(all_results)} invoices successfully!")
                    st.balloons()
    
    with tab2:
        st.header("Invoice Database")
        
        if os.path.exists(MASTER_EXCEL_FILE):
            df = pd.read_excel(MASTER_EXCEL_FILE)
            
            st.subheader(f"📊 Total Records: {len(df)}")
            
            # Search functionality
            search_col1, search_col2 = st.columns([3, 1])
            with search_col1:
                search_term = st.text_input("🔍 Search invoices...", "")
            with search_col2:
                search_field = st.selectbox("Search in", ["All", "Invoice No", "Client", "Seller"])
            
            if search_term:
                if search_field == "All":
                    mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
                else:
                    mask = df[search_field].astype(str).str.contains(search_term, case=False)
                df = df[mask]
            
            st.dataframe(df, use_container_width=True, height=400)
            
            # Statistics
            st.subheader("📈 Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Invoices", len(df))
            with col2:
                unique_clients = df['Client'].nunique() if 'Client' in df.columns else 0
                st.metric("Unique Clients", unique_clients)
            with col3:
                unique_sellers = df['Seller'].nunique() if 'Seller' in df.columns else 0
                st.metric("Unique Sellers", unique_sellers)
            
        else:
            st.info("📭 No invoices processed yet. Upload and process some invoices to see the database.")

if __name__ == "__main__":
    main()
