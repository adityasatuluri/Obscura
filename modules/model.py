from transformers import pipeline
import torch
import os

SENSITIVITY_LEVELS = {
    "Low": {"I-ACCOUNTNUM", "I-CREDITCARDNUMBER", "I-DRIVERLICENSENUM", "I-IDCARDNUM", "I-PASSWORD", "I-SOCIALNUM", "I-TAXNUM"},
    "Medium": {"I-ACCOUNTNUM", "I-CREDITCARDNUMBER", "I-DRIVERLICENSENUM", "I-IDCARDNUM", "I-PASSWORD", "I-SOCIALNUM", "I-TAXNUM",
               "I-EMAIL", "I-GIVENNAME", "I-SURNAME", "I-TELEPHONENUM", "I-USERNAME"},
    "High": {"I-ACCOUNTNUM", "I-BUILDINGNUM", "I-CITY", "I-CREDITCARDNUMBER", "I-DATEOFBIRTH", "I-DRIVERLICENSENUM", "I-EMAIL",
             "I-GIVENNAME", "I-IDCARDNUM", "I-PASSWORD", "I-SOCIALNUM", "I-STREET", "I-SURNAME", "I-TAXNUM", "I-TELEPHONENUM",
             "I-USERNAME", "I-ZIPCODE"}
}

ALL_PII_TYPES = sorted(set().union(*SENSITIVITY_LEVELS.values()))

def load_model(base_dir):
    local_model_path = os.path.join(base_dir, "models", "obscura-redact")
    print(local_model_path,"\n\n\n\n\n")
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Model directory not found at: {local_model_path}")
    try:
        return pipeline("token-classification", model=local_model_path, tokenizer=local_model_path, device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        raise Exception(f"Failed to load pipeline: {e}")

def mask_text(pipe, text, sensitivity="Medium", custom_pii_types=None, client=None, model_name=None):
    from modules.redaction import redact_text
    predictions = pipe(text)
    allowed_labels = SENSITIVITY_LEVELS.get(sensitivity, SENSITIVITY_LEVELS["Medium"]) if sensitivity != "Custom" else set(custom_pii_types)
    redacted_text = redact_text(text, predictions, allowed_labels, client, model_name)
    return redacted_text