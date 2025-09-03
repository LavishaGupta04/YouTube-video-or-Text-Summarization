import validators
import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document 
from  youtube_transcript_api import YouTubeTranscriptApi

#Streamlit 
st.set_page_config(page_title="Langchain: Summarize text from YT or Website")
st.title("Langchain: Summarize text from YT or Website")
st.subheader("Summarize URL")

with st.sidebar:
    api_key=st.text_input("Enter your GROQ api key",type='password')
if not api_key:
    st.stop()
model=ChatGroq(model='gemma2-9b-it',groq_api_key=api_key)
generic_url=st.text_input(label="URL",label_visibility='collapsed')

#prompt

template=''' 
Provide a concise summary in 300 words of the following text,
text:{text}
'''
prompt=PromptTemplate(
    input_variables=['text'],
    template=template
)

if st.button('Summarize the content'):
    if not api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a youtube or website URL.")
    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in generic_url:
                    video_id = generic_url.split("v=")[-1].split("&")[0]
                    ytt_api = YouTubeTranscriptApi()
                    transcript=ytt_api.fetch(video_id,languages=['hi', 'en'])
                    #transcript = YouTubeTranscriptApi.get_transcript(video_id=video_id)
                    text = " ".join([entry.text for entry in transcript])
                    docs = [Document(page_content=text)]
                    #loader=YoutubeLoader(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                "Chrome/115.0.0.0 Safari/537.36"})
                    docs=loader.load()

                #Create a chain
                chain=load_summarize_chain(
                    llm=model,
                    chain_type='stuff',
                    prompt=prompt
                )
                summary=chain.run(docs)

                st.success(summary)
        except Exception as e:
            st.exception(f'Exception:{e}')
            st.exception(e) 

