import pdfplumber
import csv
import os
import glob

class PDFtoCSVConverter:
    def __init__(self, input_dir, output_dir, with_classify_line = True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.pdf_files = []
        self.csv_files = []
        self.with_classify_line = with_classify_line

        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        self.pdf_files = glob.glob(os.path.join(self.input_dir, "*.pdf"))

        if not self.pdf_files:
            print("PDF-файлы не найдены.")
            return

        for pdf_file in self.pdf_files:
            rows = self.process_file(pdf_file)
            self.save_to_csv(rows, pdf_file)

        print("\nНайденные PDF-файлы:")
        for path in self.pdf_files:
            print(" -", path)

        print("\nСозданные CSV-файлы:")
        for path in self.csv_files:
            print(" -", path)

    def process_file(self, pdf_path):
        rows = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    for line in text.split('\n'):
                        line = line.strip()
                        if line:
                            if self.with_classify_line:
                                row_type = self.classify_line(line)
                                rows.append([line, row_type])
                            else:
                                rows.append([line])
        return rows

    def save_to_csv(self, rows, pdf_path):
        base_name = os.path.basename(pdf_path).replace(".pdf", ".csv")
        csv_path = os.path.join(self.output_dir, base_name)

        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["Value", "Type"])
            writer.writerows(rows)

        self.csv_files.append(csv_path)

    def classify_line(self, line):
        if self.is_heading(line):
            return "heading"
        elif self.is_table(line):
            return "table"
        else:
            return "text"

    def is_heading(self, line):
        return (
            line.isupper() and
            len(line) > 10 and
            not any(char.isdigit() for char in line)
        )

    def is_table(self, line):
        words = line.split()
        digit_count = sum(char.isdigit() for char in line)
        return (
            len(words) >= 4 and
            digit_count > len(line) * 0.25
        )

def main():
    input_pdf_dir = "../docs/22/pdf"
    output_csv_dir = "../docs/22/csv"
    converter = PDFtoCSVConverter(input_pdf_dir, output_csv_dir)
    converter.run()

if __name__ == "__main__":
    main()
