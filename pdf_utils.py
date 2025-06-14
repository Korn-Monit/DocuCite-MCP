import tempfile
from pathlib import Path

def get_file_bytes_and_name(pdf_file):
    print("DEBUG: pdf_file type:", type(pdf_file))
    print("DEBUG: pdf_file dir:", dir(pdf_file))
    print("DEBUG: pdf_file repr:", repr(pdf_file))
    if hasattr(pdf_file, "read"):
        return pdf_file.read(), Path(pdf_file.name).name
    if isinstance(pdf_file, str):
        file_path = Path(pdf_file)
        with open(file_path, "rb") as f:
            return f.read(), file_path.name
    raise ValueError("Could not extract file bytes from uploaded file.")
