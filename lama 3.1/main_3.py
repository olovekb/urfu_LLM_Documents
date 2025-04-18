import os
import csv
import fitz
from langchain_ollama.chat_models import ChatOllama

class PDFChecker:
    def __init__(self):
        self.model_name = "mistral"

    def extract_text_with_fitz(self, pdf_path):
        full_text = ""
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks")  # list of (x0, y0, x1, y1, "text", block_no, block_type)
            sorted_blocks = sorted(blocks, key=lambda b: (round(b[1]), round(b[0])))
            for block in sorted_blocks:
                text = block[4].strip()
                if text:
                    full_text += text + "\n"
        return full_text

    def analyze(self, text, prompt):
        llm = ChatOllama(model=self.model_name)
        response = llm.invoke(f"{prompt}\n\nВот содержимое таблиц из PDF:\n{text}")
        return response["message"]["content"] if isinstance(response, dict) else str(response)

    def analyze_dir(self, dir_path, prompt_path, output_dir_path=None):
        if output_dir_path is None:
            output_dir_path = dir_path

        os.makedirs(output_dir_path, exist_ok=True)

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()

        for filename in os.listdir(dir_path):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.abspath(os.path.join(dir_path, filename))
                print(f"\nОбработка {pdf_path}")

                try:
                    text = self.extract_text_with_fitz(pdf_path)
                    if not text.strip():
                        print(f"✗ Не удалось извлечь текст из {filename}")
                        continue

                    result_text = self.analyze(text, prompt)

                    print(f"\n✓ Результат:\n\n{result_text}")

                    # Сохраняем CSV, если структура похожа на таблицу
                    if "|" in result_text and result_text.count("\n") > 3:
                        lines = [line.strip() for line in result_text.split("\n") if line.strip().startswith("|")]
                        if len(lines) >= 3:
                            headers = [h.strip() for h in lines[0].strip("|").split("|")]
                            data = [
                                [cell.strip() for cell in row.strip("|").split("|")]
                                for row in lines[2:]
                            ]
                            csv_filename = os.path.splitext(filename)[0] + ".csv"
                            output_csv_path = os.path.join(output_dir_path, csv_filename)
                            with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f)
                                writer.writerow(headers)
                                writer.writerows(data)

                            print(f"✓ CSV сохранён в {output_csv_path}")
                except Exception as e:
                    print(f"\n✗ Ошибка при обработке {filename}: {e}")

def main():
    checker = PDFChecker()
    checker.analyze_dir(dir_path="../docs/22", prompt_path="./prompt.txt")

if __name__ == "__main__":
    main()