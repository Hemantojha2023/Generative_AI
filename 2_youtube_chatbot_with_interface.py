from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
import streamlit as st


load_dotenv()

model=ChatOpenAI(model='gpt-4o-mini', temperature=0.3)

parser=StrOutputParser()

video_id='lZ3bPUKo5zc'

try:
    transcript_list=YouTubeTranscriptApi.get_transcript(video_id=video_id, languages=['en'])
    transcript=" ".join(chunks['text'] for chunks in transcript_list)
except TranscriptsDisabled:
    st.error('No caption available in this video')
    
    
'''
Why do we join chunks['text']. see when we print transcript_list we see it is a list of dictionary which has keys text, start, and duration in 
each dictionary. where each text is the transcript(speech) start at the time and duration of this transcript(sentance). so we combine each
transcript(text) which contain only text not duration and start time.
'''

splitter=RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=75)
chunks=splitter.create_documents([transcript])

vector_store=FAISS.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings()
)

retriever=vector_store.as_retriever(search_kwargs={'k':3})

st.header('YouTube Transcripts Based Chatbot')

question=st.text_input('Ask me')

prompt=PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
    """,
    input_variables=['context', 'question']
)

# add all page content in the plain text of all documents in retriever
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


parallel_chain = RunnableParallel({
    'context': RunnableLambda(lambda x: x['question']) | retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

main_chain=parallel_chain | prompt | model | parser

if question:
    result=main_chain.invoke({'question' : question})
    st.write(result)
 
   

# This chatbot is ok. but it is a single-turn question answer system. but it is not multi turn chatbot it has not memory based system 