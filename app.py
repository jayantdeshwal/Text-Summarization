import validators,streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains import load_summarize_chain 
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

## Streamlit app
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Sidebar inputs
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide both the API key and URL to get started.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube or website).")
    else:
        try:
            with st.spinner("Loading content..."):
                ## Load website or YouTube data
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False,language=["en", "hi"] )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers = {
                                "User-Agent": (
                                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                                    "Chrome/129.0.6668.90 Safari/537.36"
                                ),
                                "Accept-Language": "en-US,en;q=0.9",
                                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        },
                    )
                docs = loader.load()
                # Split the YouTube transcript into smaller chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
                split_docs = text_splitter.split_documents(docs)

                ## Initialize the model **after** API key provided
                llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

                ## Summarize
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(split_docs)

                st.success(output_summary)

        except Exception as e:
            st.error(f"Exception: {e}")