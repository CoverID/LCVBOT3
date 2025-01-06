import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
from requests.exceptions import Timeout
from datetime import datetime
import toml

# Konfigurasi logging
logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Konstanta
MAX_MESSAGES = 50
MAX_INPUT_LENGTH = 500
TIMEOUT_SECONDS = 30
DOCUMENTS_FOLDER = "documents"  # Folder untuk menyimpan dokumen PDF

# Konfigurasi Streamlit
st.set_page_config(page_title="📓 LCV ASSISTANT", layout="wide")

def initialize_session_state():
    """Inisialisasi session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False

def setup_openai():
    """Setup model OpenAI"""
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        llm = ChatOpenAI(
            temperature=0.54,
            model_name="gpt-4o-mini",  # Model dengan kemampuan bahasa Indonesia yang baik
            openai_api_key=openai_api_key,
        )
        return llm
    except Exception as e:
        logging.error(f"Error setting up OpenAI: {str(e)}")
        st.error("Error initializing AI model. Please check your API key.")
        return None

def process_documents_from_folder():
    """Proses dokumen PDF dari folder documents"""
    try:
        logging.info("Starting document processing from folder")
        documents = []
        
        # Pastikan folder documents ada
        if not os.path.exists(DOCUMENTS_FOLDER):
            os.makedirs(DOCUMENTS_FOLDER)
            logging.info(f"Created documents folder: {DOCUMENTS_FOLDER}")
            return False

        # Baca semua file PDF dalam folder
        pdf_files = [f for f in os.listdir(DOCUMENTS_FOLDER) if f.endswith('.pdf')]
        if not pdf_files:
            logging.warning("No PDF files found in documents folder")
            return False

        for pdf_file in pdf_files:
            file_path = os.path.join(DOCUMENTS_FOLDER, pdf_file)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=153
        )
        splits = text_splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create vector store
        vector_store = FAISS.from_documents(splits, embeddings)
        st.session_state.vector_store = vector_store
        st.session_state.documents_processed = True
        
        logging.info("Document processing completed successfully")
        return True

    except Exception as e:
        logging.error(f"Error in document processing: {str(e)}")
        st.error(f"Error processing documents: {str(e)}")
        return False

def setup_conversation(llm, vector_store):
    """Setup conversation chain"""
    try:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            verbose=True
        )
        
        st.session_state.conversation = conversation
        logging.info("Conversation chain setup successfully")
        return True

    except Exception as e:
        logging.error(f"Error setting up conversation: {str(e)}")
        st.error("Error setting up conversation system")
        return False

def main():
    initialize_session_state()
    
    st.title("🤖 LCV Assistant")
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;'>
        <p style='font-size: 18px; margin: 0;'>
        <strong>SELAMAT DATANG !</strong> - Chatbot ini adalah Asisten yang dilatih untuk membantu AoC dalam implementasi LCV AKHLAK 2025.
        Pastikanlah Anda memiliki koneksi internet yang baik dan stabil. 
        Terimakasih atas kesabarannya menunggu chatbot siap untuk digunakan 🙏
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Proses dokumen otomatis saat aplikasi dimulai
    if not st.session_state.documents_processed:
        with st.spinner("Memproses dokumen dari folder..."):
            if process_documents_from_folder():
                st.success("Dokumen berhasil diproses!")
                
                # Setup LLM and conversation
                llm = setup_openai()
                if llm and setup_conversation(llm, st.session_state.vector_store):
                    st.success("LCV Assistant siap digunakan!")
            else:
                st.warning("Tidak ada dokumen PDF ditemukan dalam folder 'documents'. Silakan tambahkan dokumen PDF ke folder tersebut.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("✍️Tuliskan pertanyaan Anda disini secara spesifik"):
        if len(prompt) > MAX_INPUT_LENGTH:
            st.warning(f"Pertanyaan terlalu panjang. Maksimal {MAX_INPUT_LENGTH} karakter.")
            return
            
        if not st.session_state.conversation:
            st.warning("Sistem belum siap. Pastikan ada dokumen PDF dalam folder 'documents'!")
            return
            
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        # Generate AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                with st.spinner("🤔 Sedang berpikir..."):
                    enhanced_prompt = f"""Jawablah pertanyaan berikut dalam bahasa Indonesia yang baik dan benar. Jika tidak tahu, katakan: Mohon Maaf saya tidak mendapatkan hal ini dalam pelatihan saya: {prompt}"""
                    response = st.session_state.conversation.invoke(
                        {"question": enhanced_prompt},
                        timeout=TIMEOUT_SECONDS
                    )
                    ai_response = response["answer"]
                    
                message_placeholder.write(ai_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": ai_response}
                )
                
                # Limit message history
                if len(st.session_state.messages) > MAX_MESSAGES:
                    st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]
                    
            except Timeout:
                message_placeholder.error("Waktu permintaan habis. Silakan coba lagi.")
                logging.error("Request timeout occurred")
            except openai.error.APIError as e:
                message_placeholder.error(f"OpenAI API returned an API Error: {str(e)}")
                logging.error(f"Error generating response: {str(e)}")
            except openai.error.APIConnectionError as e:
                message_placeholder.error(f"Failed to connect to OpenAI API: {str(e)}")
                logging.error(f"Connection error: {str(e)}")
            except Exception as e:
                message_placeholder.error(f"Terjadi kesalahan: {str(e)}")
                logging.error(f"Error generating response: {str(e)}")

    # Disclaimer
    st.markdown(
        """
        <p style='font-size: 12px; font-style: italic; color: gray;'>
        ⚠️ <strong>Disclaimer:</strong> AI-LLM model dapat saja membuat kesalahan. CEK KEMBALI INFO PENTING.
        </p>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
