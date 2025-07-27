import os
from dotenv import load_dotenv
import easyocr
from PIL import Image
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import json_loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


load_dotenv()
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    print("Google API Key loaded successfully.")
except Exception as e:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.") from e

def model():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key, temperature=0.3)
    print("Model initialized successfully.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)
    return model, embeddings

def prepare_data():
    loader = json_loader.JSONLoader(file_path="data/incidecoder_data.json", jq_schema=".[]", text_content=False)
    data = loader.load()
    print(f"Loaded {len(data)} documents from JSON file.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    print(f"Split documents into {len(docs)} chunks.")
    return docs

def create_vectore_db(docs, embeddings):
    vector_db = Chroma.from_documents(documents=docs, embedding=embeddings)
    print(f"Vector database created with {len(vector_db)} documents.")
    print("Embeddings initialized successfully.")
    return vector_db

def image_processing(image_path):
    reader = easyocr.Reader(['en'], gpu=False)
    image = Image.open(image_path)
    image = np.array(image)
    result = reader.readtext(image, detail=0)
    return result


def create_qa_chain(model, vector_db):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template = """
        You are a SkinCare expert with extensive knowledge about skincare and beauty. Your primary function is to respond to 
        user inquiries in a clear and understandable manner. 
        Use only the information provided in the context below to generate your response. Do not rely on prior knowledge. 
        If the answer cannot be found in the context, respond with: "I do not have the information right now."
        If a user provides a list of ingredients, analyze each ingredient and provide a detailed description, including their 
        benefits, features, potential allergens, and any other relevant information - only if this information is present in the context. 
        If a user uploads an image of a product, analyze the ingredients present in the image using the provided context. 
        If the image is unclear or incorrect, kindly prompt the user to upload a correct image for accurate analysis. 
        Do not make up information. Stay strictly within what is present in the context.

        Context:
        {context}

        Question:
        {question}

        Answer:"""
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    print("QA chain created successfully.")
    return qa_chain

def main():
    model_instance, embeddings = model()
    docs = prepare_data()
    vector_db = create_vectore_db(docs, embeddings)
    qa_chain = create_qa_chain(model_instance, vector_db)

    # query = "What is the use of Diisopropyl Dimer Dilinoleate?"
    while True:
        print("\nType 'exit' to quit the application.")
        input_query = input("Get started with your Question:")
        if input_query.lower() == 'exit':
            print("Exiting the application.")
            break
        elif input_query.strip() == "":
            print("Please enter a valid query.")
            continue
        elif input_query.endswith('.png') or input_query.endswith('.jpg'):
            query = " ".join(image_processing(input_query))
        else:
            query = input_query
        # image = "test_image1.png"
        # result = image_processing(image)
        # query = " ".join(result)
        result = qa_chain.invoke({"query": query})

        print("Answer:", result['result'])
        # print("Source Documents:", [doc.metadata for doc in result['source_documents']])

if __name__ == "__main__":
    main()