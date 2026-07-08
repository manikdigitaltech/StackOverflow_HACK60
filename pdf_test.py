from pypdf import PdfReader
from pathlib import Path

# PDF file path
pdf_path = Path("data/sample_paper.pdf")

# Check if PDF exists

if not pdf_path.exists():
    print("PDF file not found.")
    print("Please put your PDF inside the data folder with this name:")
    print("data/sample_paper.pdf")
    exit()

# Read PDF
reader = PdfReader(pdf_path)

print("PDF loaded successfully.")
print("Total pages:", len(reader.pages))

# Extract text from first 3 pages only
text = ""

for page_number, page in enumerate(reader.pages[:3], start=1):
    page_text = page.extract_text()

    if page_text:
        text += f"\n\n--- Page {page_number} ---\n"
        text += page_text

# Print first 2000 characters
print("\nExtracted text preview:")
print(text[:2000])

# Save extracted text into a text file
output_path = Path("data/extracted_text.txt")

with open(output_path, "w", encoding="utf-8") as file:
    file.write(text)

print("\nText extraction completed.")
print("Saved extracted text at:")
print(output_path)