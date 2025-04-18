import os
import csv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain.chains import RetrievalQA

class PDFChecker:
    def __init__(self):
        self.model_name = "llama3.2"
        self.embed_model = "nomic-embed-text"

    def analyze(self, pdf_path, prompt):
        pages = self.load_pdf(pdf_path)
        db = self.create_vector_db(pages)
        llm = ChatOllama(model=self.model_name)
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

        return qa.invoke(prompt)

    def analyze_dir(self, dir_path, prompt_path, output_dir_path = None):
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
                    result = self.analyze(pdf_path, prompt)
                    result_text = result["result"] if isinstance(result, dict) and "result" in result else str(result)

                    print(f"\n✓ Результат:\n\n{result_text}")

                    if "|" in result_text and result_text.count("\n") > 3:
                        lines = [line.strip() for line in result_text.split("\n") if line.strip().startswith("|")]
                        if len(lines) >= 3:
                            headers = [h.strip() for h in lines[0].strip("|").split("|")]
                            data = [
                                [cell.strip() for cell in row.strip("|").split("|")]
                                for row in lines[2:]
                            ]
                            csv_filename = os.path.splitext(filename)[0] + ".csv"
                            output_path = os.path.abspath(os.path.join(output_dir_path, csv_filename))
                            with open(output_path, "w", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f)
                                writer.writerow(headers)
                                writer.writerows(data)

                            print(f"\n✓ Результат сохранён в {output_path}")
                except Exception as e:
                    print(f"\n✗ Ошибка при обработке {filename}: {e}")

    def load_pdf(self, pdf_path):
        loader = PyPDFLoader(pdf_path)

        return loader.load_and_split()

    def create_vector_db(self, pages):
        embedding = OllamaEmbeddings(model=self.embed_model)

        return FAISS.from_documents(pages, embedding)

def main():
    checker = PDFChecker()
    checker.analyze_dir(dir_path="../docs/22", prompt_path="./prompt.txt")

if __name__ == "__main__":
    main()