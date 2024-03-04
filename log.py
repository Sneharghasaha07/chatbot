from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings;from langchain.chains import ConversationalRetrievalChain;from langchain.chat_models import ChatOpenAI;from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from decouple import config

embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="../vector_db",
    collection_name="indian_oil",
    embedding_function=embedding_function,
)


# create prompt
QA_prompt = PromptTemplate(
    template="""Use the following pieces of context to answer the user question.
chat_history: {chat_history}
Context: {text}
Question: {question}
Answer:""",
    input_variables=["text", "question", "chat_history"]
)

llm = ChatOpenAI(openai_api_key=config("kkk"), temperature=0)

# create memory
memory = ConversationBufferMemory(
    return_messages=True, memory_key="chat_history")

# create retriever chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=vector_db.as_retriever(
        search_kwargs={'fetch_k': 4, 'k': 3}, search_type='mmr'),
    chain_type="refine",
)

# question
question = "tell me about iocl?"

# call QA chain
response = qa_chain({"question": question})


print(response.get("answer"))