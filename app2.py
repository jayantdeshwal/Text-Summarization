import validators, streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains import load_summarize_chain, LLMChain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

## Streamlit app
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Sidebar inputs
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

## Summarization prompt
prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

## Translation prompt
translate_prompt_template = """
Translate the following text into English. Preserve meaning and context:
Text: {text}
"""
translate_prompt = PromptTemplate(template=translate_prompt_template, input_variables=["text"])

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
                    try:
                        # Try English first
                        loader = YoutubeLoader.from_youtube_url(
                            generic_url,
                            add_video_info=False,
                            language=["en"]
                        )
                        docs = loader.load()
                    except Exception:
                        # Fallback to Hindi if English not found
                        try:
                            loader = YoutubeLoader.from_youtube_url(
                                generic_url,
                                add_video_info=False,
                                language=["hi"],
                                proxies={"http": "http://proxy_ip:port", "https": "http://proxy_ip:port"}
                            )
                            docs = loader.load()
                        except Exception as e:
                            st.error(f"Could not retrieve a YouTube transcript: {e}")
                            st.stop()
                else:
                    # Load website content
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
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

                # Split the text into manageable chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
                split_docs = text_splitter.split_documents(docs)

                ## Initialize the model after API key provided
                llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

                # Create the translation chain once
                translation_chain = LLMChain(llm=llm, prompt=translate_prompt)
                ## Translate non-English transcripts to English
                translated_docs = []
                for doc in split_docs:
                    translated_text = translation_chain.run({"text": doc.page_content})
                    doc.page_content = translated_text
                    translated_docs.append(doc)

                ## Summarize translated text
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(translated_docs)

                st.success(output_summary)

        except Exception as e:
            st.error(f"Exception: {e}")
