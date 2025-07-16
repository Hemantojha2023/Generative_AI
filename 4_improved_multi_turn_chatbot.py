from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory
import gradio as gr

load_dotenv()

# Load and split PDF
loader = PyPDFLoader('the_nestle_hr_policy_pdf_2012.pdf')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
chunks = splitter.split_documents(docs)

# Vector store and retriever
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    collection_name='nestle'
)
retriever = vector_store.as_retriever()

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer ONLY from the provided transcript context. "
               "If the context is insufficient, just say you don't know."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
    ("system", "Context:\n{context}")    # retrieve relevant context text(answer) from vector_store
])

# Model and parser
model = ChatOpenAI(model='gpt-4')
parser = StrOutputParser()

# Use ChatMessageHistory to keep the chat as memory
chat_history = ChatMessageHistory()

# Format retrieved docs
def format_docs(all_docs):
    return "\n\n".join(doc.page_content for doc in all_docs)

# Combine retriever & history
parallel_chain = RunnableParallel({
    'context': RunnableLambda(lambda x: x['question']) | retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough(),
    'chat_history': RunnableLambda(lambda x: chat_history.messages)  # direct messages to prompt(chatprompt) that stored message in chat_history
})   # See the structure or graph of parallel chain in notebook(copy)

# Final chain (See the structure or graph of parallel chain and final chain in notebook(copy))
final_chain = parallel_chain | prompt | model | parser

# Gradio function
def chatbot_fn(message, history):
    answer = final_chain.invoke({'question': message})
    # Update the history manually
    chat_history.add_user_message(message)
    chat_history.add_ai_message(answer)
    return answer

# Launch Gradio
gr.ChatInterface(
    fn=chatbot_fn,
    title="Nestle HR Policy Chatbot (modern memory)",
    description="Ask questions about the Nestle HR Policy PDF. Multi-turn memory included!",
    chatbot=gr.Chatbot(type="messages"),
    type="messages"
).launch()    # use share=True in launch for public link