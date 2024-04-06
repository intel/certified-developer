from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceEmbeddings
from datasets import load_dataset

import pandas as pd
import time
import os

class PickerBot():
    
    def __init__(self, data, model):
        self.data = data
        self.model = model
        
        
    def data_proc(self):
        
        if not os.path.isfile(self.data): 
            # Download the customer service robot support dialogue from hugging face
            dataset = load_dataset("FunDialogues/customer-service-apple-picker-maintenance", cache_dir=None)

            # Convert the dataset to a pandas dataframe
            dialogues = dataset['train']
            df = pd.DataFrame(dialogues, columns=['id', 'description', 'dialogue'])

            # Print the first 5 rows of the dataframe
            df.head()

            # only keep the dialogue column
            dialog_df = df['dialogue']
            
            # save the data to txt file
            dialog_df.to_csv(self.data, sep=' ', index=False)
        else:
            print('data already exists in path.')

    def create_vectorstore(self, chunk_size: int = 500, overlap: int = 25):
        loader = TextLoader(self.data)
        # Text Splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        # Embed the document and store into chroma DB
        self.index = VectorstoreIndexCreator(embedding= HuggingFaceEmbeddings(), text_splitter=text_splitter).from_loaders([loader])


    def inference(self, user_input: str, context_verbosity: bool = False, top_k: int=2):
        # perform similarity search and retrieve the context from our documents
        results = self.index.vectorstore.similarity_search(user_input, k=top_k)
        # join all context information into one string 
        context = "\n".join([document.page_content for document in results])
        if context_verbosity:
            print(f"Retrieving information related to your question...")
            print(f"Found this content which is most similar to your question: {context}")

        template = """
        Please use the following apple picker technical support related questions to answer questions. 
        Context: {context}
        ---
        This is the user's question: {question}
        Answer: This is what our auto apple picker technical expert suggest."""

        prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)

        llm_chain = LLMChain(prompt=prompt, llm=self.model)
        
        print("Processing the information with gpt4all...\n")
        start_time = time.time()
        response = llm_chain.run(user_input)
        elapsed_time_milliseconds  = (time.time() - start_time) * 1000
        
        tokens = len(response.split())
        time_per_token_milliseconds = elapsed_time_milliseconds  / tokens if tokens != 0 else 0
        
        processed_reponse = response + f" --> {time_per_token_milliseconds:.4f} milliseconds/token AND Time taken for response: {elapsed_time_milliseconds:.2f} milliseconds"
        
        return processed_reponse