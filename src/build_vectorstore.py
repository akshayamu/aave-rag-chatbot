import time
from document_processor import load_and_process_documents, create_vectorstore, save_vectorstore

def main():
    print("Building vectorstore from Aave documentation...")
    start_time = time.time()

    documents = load_and_process_documents()
    if not documents:
        print("No documents found!")
        return
    print(f"Loaded {len(documents)} document chunks")

    vectorstore = create_vectorstore(documents)
    save_vectorstore(vectorstore)
    print(f"Vectorstore saved in {time.time() - start_time:.2f} seconds!")

if __name__ == "__main__":
    main()
