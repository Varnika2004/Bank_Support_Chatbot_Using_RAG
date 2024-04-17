# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Importing the dataset
import pandas as pd
dataset = pd.read_csv('bank_faq.csv')
# print(dataset.head())

# Extracting the questions
questions = dataset.iloc[:, 0]
# print(questions.head())

# Making documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000)
docs = text_splitter.create_documents(questions)
print(len(docs))

# Creating embedding for similarity search
# Parameters
model_path = 'sentence-transformers/all-mpnet-base-v2'
encoder_args = {'device': 'cpu'}
encoder_test = {'normalize_embeddings': False}

# Creation of Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name = model_path,
                              model_kwargs = encoder_args,
                              encode_kwargs = encoder_test)

print("hai")

# Creating the FAISS DB
from langchain.vectorstores import FAISS
db = FAISS.from_documents(docs, embeddings)

# # Dumping the faiss file
# import joblib
# joblib.dump(db, 'db.faiss')



from langchain.vectorstores import FAISS
import pandas as pd
import joblib
import numpy as np


# Retriving the db
db = joblib.load('db.faiss')
dataFrame = pd.read_csv('bank_faq.csv')

# The streamlit application
import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(_name_)

def run():
  st.title("🏦*AnnAI **Chat Companion*")

  if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, I'm here to guide you on precised decisions"}]

  for message in st.session_state.messages:
    with st.chat_message(message['role']):
      st.write(message["content"])

  prompt = st.chat_input("Felt confused ? Feel free to query me 😊")

  if prompt: # if input is given
    # updating user input to the chat container
    with st.chat_message("user"):
      st.markdown(f"Me: {prompt}")

    # appending to the chat history 
    st.session_state.messages.append({"role": "user", "content": prompt})


    # Searching for similarity in vector_DB to extract question
    search_doc = db.similarity_search(prompt)[0]
    search_doc_content = search_doc.page_content

    # Searching the dataset for corresponding answer
    corresponding_data = dataFrame[dataFrame['Question'] == str(search_doc_content)]

    # Updating the chat_container and history_chat
    with st.chat_message("assistant"):
        if corresponding_data.empty == False:
            chat_result = corresponding_data.iloc[0, 1]
            st.markdown(f"DocBot: {chat_result}")

        else:
           chat_result = "Hai, nice to hear you"
           st.markdown(f"DocBot: {chat_result}")

    # Appending the chat history list
    st.session_state.messages.append({"role": "assistant", "content": chat_result})


if _name_ == "_main_":
    run()


# To run this app, use:
#   streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false