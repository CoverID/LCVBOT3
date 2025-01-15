import os
import sys
import subprocess
import streamlit as st
import torch
import toml

# Import langchain packages
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Konfigurasi halaman
st.set_page_config(
    page_title="LCV-ASSISTANT",
    page_icon="🤖",
    layout="wide"
)

# CSS Styling
st.title("🤖 LCV-ASSISTANT")
st.markdown(
    """
    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;'>
    <p style='font-size: 18px; margin: 0;'>
    <strong>Selamat datang !</strong> - Chatbot ini adalah Asisten yang dilatih untuk membantu AoC dalam implementasi LCV AKHLAK 2025.
    Pastikanlah Anda memiliki koneksi internet yang baik dan stabil. Terimakasih atas kesabarannya menunggu chatbot siap untuk digunakan 🙏
    </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Custom prompt template dengan optimasi untuk Bahasa Indonesia
template = """Anda adalah asisten AI yang ahli dalam dokumen berbahasa Indonesia.
Gunakan konteks berikut untuk menjawab pertanyaan dengan:
1. Bahasa Indonesia yang baik, formal, dan terstruktur
2. Referensi ke bagian dokumen yang relevan jika ada
3. Penjelasan yang jelas dan terorganisir

Konteks: {context}
Riwayat Chat: {chat_history}
Pertanyaan: {question}

Instruksi khusus:
- Jika informasi tidak ditemukan dalam konteks, katakan "Mohon maaf, saya tidak memiliki informasi tersebut dalam dokumen yang tersedia."
- Berikan jawaban yang ringkas namun komprehensif
- Jika ada istilah teknis, berikan penjelasan singkatnya

Jawaban:"""

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=template
)

# Fungsi preprocessing untuk dokumen bahasa Indonesia
def preprocess_document(text: str) -> str:
    """Preprocessing khusus untuk dokumen Bahasa Indonesia"""
    import re

    # Bersihkan karakter khusus
    text = re.sub(r'[^\w\s\.]', ' ', text)

    # Normalisasi spasi
    text = ' '.join(text.split())

    # Handling untuk singkatan umum bahasa Indonesia
    abbreviations = {
        'yg': 'yang',
        'dgn': 'dengan',
        'utk': 'untuk',
        'tsb': 'tersebut',
        'dll': 'dan lain-lain',
        'dst': 'dan seterusnya',
        'spt': 'seperti',
        'krn': 'karena',
        'pd': 'pada',
        'dr': 'dari',
        'knp': 'kenapa',
        'LCV': 'Living Core Values',
        'PCB': 'Project Charter Budaya'
    }

    for abbr, full in abbreviations.items():
        text = re.sub(r'\b' + abbr + r'\b', full, text, flags=re.IGNORECASE)

    return text

# Konstanta
DOCUMENTS_PATH = "documents"
MODEL_NAME = "gpt-4o-mini"  # Menggunakan model yang paham bahasa indonesia
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MAX_TOKENS = 1500
TEMPERATURE = 0.45

# Pustaka Data
PUSTAKA_DATA = {
    "files": [
        {
            "title": "9 Parameter 2025",
            "description": "9 Parameter Penilaian LCV AKHLAK 2025",
            "url": "https://drive.google.com/file/d/11DOL9kG0ttp0ilJ_5ykLrSjKVEI9IJlV/view?usp=sharing"
        },
        {
            "title": "10 Fokus Keberlanjutan Pertamina",
            "description": "Penjelasan 10 Fokus Keberlanjutan Pertamina beserta contohnya.",
            "url": "https://drive.google.com/file/d/1FTIttFp17nGh5Pfc_w-wS-Xf7D_aLUrg/view?usp=sharing"
        },
        {
            "title": "Contoh PCB dengan Allignment terhadap 10 Fokus Keberlanjutan Pertamina",
            "description": "Berbagai contoh PCB yang memiliki program budaya menyasar pada 10 Fokus Keberlnajutan Pertamina.",
            "url": "https://drive.google.com/file/d/17Bx_ha1o01UsrovI6TVadmupP9LxN-J5/view?usp=sharing"
        },
        {
            "title": "Contoh Klasifikasi Program",
            "description": "Contoh-contoh klasifikasi program: Strategis, Taktikal, Operasional",
            "url": "https://docs.google.com/spreadsheets/d/1irDS2zSD8yavfEf5uLDSpuCY65T_UIe0/edit?usp=sharing"
        },
        {
            "title": "Form Kuantifikasi Impact to Business",
            "description": "Form kuantifikasi impact to business dan contoh pengisian",
            "url": "https://docs.google.com/spreadsheets/d/1W2jlrIhiJac_1oLd86dSSXLMihgV1KO4/edit?usp=sharing"
        },
        {
            "title": "Sosialisasi LCV 2025",
            "description": "Materi sosialisasi LCV 2025",
            "url": "https://drive.google.com/file/d/1iXtwOtd0BCF4tQnz2R2Obek3I2E0h5cM/view?usp=sharing"
        },
        {
            "title": "Dashboard PowerBI",
            "description": "Nilai Kualitatif Evidence Bulanan",
            "url": "https://ptm.id/skorlivingcorevaluesAKHLAK"
        },
        {
            "title": "Konfirmasi Evidence",
            "description": "Form Konfirmasi Evidence LCV 2025",
            "url": "https://ptm.id/FormKonfirmasiEvidenceLCV2025"
        }
    ]
}

def silent_install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q", "--disable-pip-version-check"])
        return True
    except Exception:
        return False

def check_and_install_packages():
    required_packages = [
        'langchain_openai',
        'transformers',
        'PyPDF2',
        'sentence-transformers',
        'faiss-cpu',
        'torch',
        'langchain_community',
        'langchain',
        'toml'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        with st.spinner('🔧 Mempersiapkan sistem...'):
            for package in missing_packages:
                silent_install_package(package)

    return True

def display_pustaka():
    st.sidebar.title("Referensi Penunjang")
    st.sidebar.write("Daftar Referensi yang Tersedia:")

    for file in PUSTAKA_DATA['files']:
        st.sidebar.write(f"**{file['title']}**")
        st.sidebar.write(f"Deskripsi: {file['description']}")
        st.sidebar.markdown(f"[Buka Link]({file['url']})")
        st.sidebar.write("---")

def load_api_key():
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            return api_key

        with open('.streamlit/secrets.toml', 'r') as f:
            config = toml.load(f)
            return config.get('OPENAI_API_KEY')
    except FileNotFoundError:
        st.error("File secrets.toml tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Error membaca API key: {str(e)}")
        return None

@st.cache_resource
def load_embeddings_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return HuggingFaceEmbeddings(
        model_name="LazarusNLP/all-indo-e5-small-v4",
        model_kwargs={
            'device': device
        }
    )

def process_documents():
    try:
        if not os.path.exists(DOCUMENTS_PATH):
            os.makedirs(DOCUMENTS_PATH)
            st.warning(f"Folder {DOCUMENTS_PATH} telah dibuat. Silakan tambahkan file PDF Anda.")
            return None

        pdf_files = [f for f in os.listdir(DOCUMENTS_PATH) if f.endswith('.pdf')]
        if not pdf_files:
            st.warning("Tidak ada file PDF yang ditemukan dalam folder documents")
            return None

        documents = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(os.path.join(DOCUMENTS_PATH, pdf_file))
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]                  
        )

        chunks = text_splitter.split_documents(documents)
        embeddings = load_embeddings_model()
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store

    except Exception as e:
        st.error(f"Error dalam memproses dokumen: {str(e)}")
        return None

def get_conversation_chain(vector_store, api_key):
    try:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer',
            max_token_limit=2000
        )

        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            presence_penalty=0.1,
            frequency_penalty=0.1,
            top_p=0.95
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={'prompt': CUSTOM_PROMPT}
        )

        return conversation_chain
    except Exception as e:
        st.error(f"Gagal membuat conversation chain: {str(e)}")
        return None

def main():
    if not check_and_install_packages():
        st.error("Gagal menyiapkan sistem")
        return

    api_key = load_api_key()
    if not api_key:
        st.error("API key tidak ditemukan. Pastikan file secrets.toml berisi OPENAI_API_KEY")
        return

    if 'vector_store' not in st.session_state:
        with st.spinner('Mempersiapkan sistem...'):
            vector_store = process_documents()
            if vector_store is None:
                return
            st.session_state.vector_store = vector_store
            st.session_state.conversation = get_conversation_chain(vector_store, api_key)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("✍️ Tuliskan pertanyaan Anda disini secara spesifik"):
        if not prompt.strip():
            st.warning("Mohon masukkan pertanyaan yang valid")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                modified_prompt = "Berikan jawaban dalam bahasa Indonesia yang baik dan terstruktur: " + prompt
                with st.spinner("🤔 Sedang berpikir..."):
                    response = st.session_state.conversation({"question": modified_prompt})
                    st.markdown(response['answer'])
                    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
            except Exception as e:
                error_message = "Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi."
                st.error(f"{error_message}\nDetail error: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Footer
st.markdown("""
---
**Disclaimer:**
- Sistem ini menggunakan AI-LLM dan dapat menghasilkan jawaban yang tidak selalu akurat
- Untuk mengurangi halusinasi dan meningkatkan akurasi, gunakanlah sistem prompt R-A-G :
  [Role] Anda adalah seorang ahli Living Core Values AKHLAK yang kreatif.
  [Action] Tolong berikan 3 buah ide  program ONE KOLAB yang strategis,
  [Goals] dengan tujuan menciptakan safety dan produktivitas yang lebih baik.
- Mohon verifikasi informasi penting dengan sumber terpercaya
""")

if __name__ == "__main__":
    try:
        display_pustaka()
        main()
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
