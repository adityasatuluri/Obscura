import streamlit as st
from modules.model import load_model, mask_text, SENSITIVITY_LEVELS, ALL_PII_TYPES
from modules.redaction import redact_pdf, redact_txt, redact_xlsx, redact_docx, redact_image
import os
import pandas as pd
import torch
from groq import Groq

# Base directory and upload folder
base_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(base_dir, "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load NER model
pipe = load_model(base_dir)

# Streamlit UI
st.title("EDR Tool")

# Sidebar for Sensitivity, PII Types, and Groq API Settings
with st.sidebar:
    st.header("EDR Tool Settings")
    sensitivity = st.selectbox("Sensitivity Level", options=["Low", "Medium", "High", "Custom"], index=1)
    
    if sensitivity == "Custom":
        custom_pii_types = st.multiselect("Select PII Types to Redact (All the PFI, PHI types will be redacted.)", options=ALL_PII_TYPES, default=ALL_PII_TYPES[:5])
        if not custom_pii_types:
            st.warning("Please select at least one PII type for custom sensitivity.")
    else:
        custom_pii_types = None
        # st.write("**PII Types Included:**")
        # for pii_type in sorted(SENSITIVITY_LEVELS[sensitivity]):
        #     st.write(f"- {pii_type}")
    
    st.info(f"Running on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Groq API Settings
    st.header("Groq API Settings")
    groq_api_key = st.text_input("Groq API Key", type="password", value="")
    groq_model = st.selectbox(
        "Groq Model",
        options=["llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it", "llama3-70b-8192"],
        index=0
    )
    
    # Usage Limits Expander
    with st.expander("Usage Limits for Selected Models", expanded=False):
        usage_limits = {
            "llama3-8b-8192": {"RPM": 30, "RPD": 14400, "TPM": 6000, "TPD": 500000},
            "mixtral-8x7b-32768": {"RPM": 30, "RPD": 14400, "TPM": 5000, "TPD": 500000},
            "gemma2-9b-it": {"RPM": 30, "RPD": 14400, "TPM": 15000, "TPD": 500000},
            "llama3-70b-8192": {"RPM": 30, "RPD": 14400, "TPM": 6000, "TPD": 500000}
        }
        for model, limits in usage_limits.items():
            st.write(f"**{model}**")
            st.write(f"- Requests per Minute (RPM): {limits['RPM']}")
            st.write(f"- Requests per Day (RPD): {limits['RPD']}")
            st.write(f"- Tokens per Minute (TPM): {limits['TPM']}")
            st.write(f"- Tokens per Day (TPD): {limits['TPD']}")
            st.write("")  # Add a blank line for readability
    st.info("Get your Groq API key from console.groq.com.")

# Initialize Groq client if API key is provided
client = None
if groq_api_key:
    try:
        client = Groq(api_key=groq_api_key)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {str(e)}")

# Main Content
tab1, tab2 = st.tabs(["Text Input", "File Upload"])

# Text Input Tab
with tab1:
    st.subheader("Redact Text")

    text = """
                Hey there! I’d like to introduce you to John Doe, a wonderful employee with the employee ID 12345. John was born on January 1, 1990, and currently lives at 123 Main Street, Anytown, CA 91234. You can reach out to him via email at john.doe@gmail.com or give him a call at (555) 123-4567. Now, when it comes to his sensitive details, his Social Security Number is 123-45-1234, his Driver’s License Number is CA12345678, and his Bank Account Number is 112233445566—definitely examples of Personally Identifiable Information (PII) and Protected Financial Information (PFI) that we need to handle with care! Oh, and his Passport Number is A1234567. On the health side, John is also associated with a Hospital ID of 3456782 and a Patient ID of 989, and unfortunately, he’s been diagnosed with cancer—information that falls under Protected Health Information (PHI) and requires strict confidentiality. Let’s make sure we protect all this data appropriately!

                Diving a bit deeper into John’s health journey, he’s been managing his condition with a few prescribed medications. His doctor at the hospital has him on a daily dose of 50 mg of Capecitabine, a chemotherapy drug, and he also takes 10 mg of Ondansetron to help with nausea as a side effect. On top of that, he’s been scheduled for a follow-up MRI scan next month, under his medical record number MRN-456789, to monitor his progress. All of this, including his medication list and upcoming procedures, is sensitive Protected Health Information (PHI) that we need to keep secure and only share with authorized medical personnel. It’s really important to support John while respecting his privacy during this challenging time!

                On the financial side, John has been keeping his affairs in order too. Besides his bank account, he has a savings account with the number 998877665544 at First National Bank, and he recently opened a credit card with the number ending in 7890—both of which are Protected Financial Information (PFI) that we need to safeguard to prevent identity theft or fraud. He also mentioned that he’s been working with a financial advisor to manage a small investment portfolio, which includes a brokerage account numbered BRK-123456. It’s great to see John staying on top of his finances, but we’ll make sure all this financial data stays protected and confidential, just like his health and personal info!
            """

    input_text = st.text_area("Enter text to mask PII",
                              text,
                              height=150)
    
    if st.button("Mask Text"):
        if not input_text:
            st.error("Please enter some text to mask.")
        elif sensitivity == "Custom" and not custom_pii_types:
            st.error("Please select at least one PII type for custom sensitivity.")
        elif not groq_api_key:
            st.error("Please provide a Groq API key in the sidebar.")
        else:
            with st.spinner("Processing..."):
                try:
                    redacted_text = mask_text(pipe, input_text, sensitivity, custom_pii_types, client=client, model_name=groq_model)
                    # Split the redacted text into lines
                    lines = redacted_text.splitlines()
                    # Check if the first line contains "redacted text" (case-insensitive)
                    if lines and "redacted text" in lines[0].lower():
                        # Remove the first line
                        lines.pop(0)
                    # Join the lines back together
                    processed_text = "\n".join(lines)
                    # Display the processed text
                    st.text_area("Redacted Output", processed_text, height=150, disabled=True)
                    # Add download button for the processed text
                    st.download_button(
                        label="Download Redacted Text",
                        data=processed_text,
                        file_name="redacted_text.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}. Falling back to masking without generative AI.")
                    redacted_text = mask_text(pipe, input_text, sensitivity, custom_pii_types, client=None, model_name=None)
                    # Split the redacted text into lines
                    lines = redacted_text.splitlines()
                    # Check if the first line contains "redacted text" (case-insensitive)
                    if lines and "redacted text" in lines[0].lower():
                        # Remove the first line
                        lines.pop(0)
                    # Join the lines back together
                    processed_text = "\n".join(lines)
                    # Display the processed text
                    st.text_area("Redacted Output", processed_text, height=150, disabled=True)
                    # Add download button for the processed text
                    st.download_button(
                        label="Download Redacted Text",
                        data=processed_text,
                        file_name="redacted_text.txt",
                        mime="text/plain"
                    )

# File Upload Tab
with tab2:
    st.subheader("Redact File")
    uploaded_file = st.file_uploader("Upload a file (PDF, TXT, XLSX, DOCX, Image)", type=["pdf", "txt", "xlsx", "docx", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        if sensitivity == "Custom" and not custom_pii_types:
            st.error("Please select at least one PII type for custom sensitivity.")
        elif not groq_api_key:
            st.error("Please provide a Groq API key in the sidebar.")
        else:
            with st.spinner("Processing file..."):
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                try:
                    if file_ext == '.pdf':
                        pdf_output, redacted_text = redact_pdf(pipe, uploaded_file, sensitivity, custom_pii_types, client=client, model_name=groq_model)
                        st.text_area("Redacted PDF Text", redacted_text, height=150, disabled=True)
                        st.download_button("Download Redacted PDF", pdf_output, file_name=f"redacted_{uploaded_file.name}", mime="application/pdf")
                    
                    elif file_ext == '.txt':
                        redacted_text = redact_txt(pipe, uploaded_file, sensitivity, custom_pii_types, client=client, model_name=groq_model)
                        st.text_area("Redacted TXT", redacted_text, height=150, disabled=True)
                    
                    elif file_ext == '.xlsx':
                        csv_content = redact_xlsx(pipe, uploaded_file, sensitivity, custom_pii_types, client=client, model_name=groq_model)
                        st.dataframe(pd.read_csv(csv_content))
                        st.download_button("Download Redacted CSV", csv_content.getvalue(), file_name=f"redacted_{os.path.splitext(uploaded_file.name)[0]}.csv", mime="text/csv")
                    
                    elif file_ext == '.docx':
                        docx_output, redacted_text = redact_docx(pipe, uploaded_file, sensitivity, custom_pii_types, client=client, model_name=groq_model)
                        st.text_area("Redacted DOCX Text", redacted_text, height=150, disabled=True)
                        st.download_button("Download Redacted DOCX", docx_output, file_name=f"redacted_{uploaded_file.name}", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                    
                    elif file_ext in ('.png', '.jpg', '.jpeg'):
                        image_output = redact_image(pipe, uploaded_file, sensitivity, custom_pii_types, client=client, model_name=groq_model)
                        st.image(image_output, caption="Redacted Image")
                        st.download_button("Download Redacted Image", image_output, file_name=f"redacted_{uploaded_file.name}", mime="image/png")
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}. Falling back to masking without generative AI.")
                    if file_ext == '.pdf':
                        pdf_output, redacted_text = redact_pdf(pipe, uploaded_file, sensitivity, custom_pii_types, client=None, model_name=None)
                        st.text_area("Redacted PDF Text", redacted_text, height=150, disabled=True)
                        st.download_button("Download Redacted PDF", pdf_output, file_name=f"redacted_{uploaded_file.name}", mime="application/pdf")
                    elif file_ext == '.txt':
                        redacted_text = redact_txt(pipe, uploaded_file, sensitivity, custom_pii_types, client=None, model_name=None)
                        st.text_area("Redacted TXT", redacted_text, height=150, disabled=True)
                    elif file_ext == '.xlsx':
                        csv_content = redact_xlsx(pipe, uploaded_file, sensitivity, custom_pii_types, client=None, model_name=None)
                        st.dataframe(pd.read_csv(csv_content))
                        st.download_button("Download Redacted CSV", csv_content.getvalue(), file_name=f"redacted_{os.path.splitext(uploaded_file.name)[0]}.csv", mime="text/csv")
                    elif file_ext == '.docx':
                        docx_output, redacted_text = redact_docx(pipe, uploaded_file, sensitivity, custom_pii_types, client=None, model_name=None)
                        st.text_area("Redacted DOCX Text", redacted_text, height=150, disabled=True)
                        st.download_button("Download Redacted DOCX", docx_output, file_name=f"redacted_{uploaded_file.name}", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                    elif file_ext in ('.png', '.jpg', '.jpeg'):
                        image_output = redact_image(pipe, uploaded_file, sensitivity, custom_pii_types, client=None, model_name=None)
                        st.image(image_output, caption="Redacted Image")
                        st.download_button("Download Redacted Image", image_output, file_name=f"redacted_{uploaded_file.name}", mime="image/png")