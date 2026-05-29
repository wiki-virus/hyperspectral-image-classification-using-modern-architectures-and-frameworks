from docx import Document

doc = Document("Efficiency and Accuracy of Few.docx")
for i, para in enumerate(doc.paragraphs):
    if para.text.strip():
        print(f"[{i}] {para.text}")
