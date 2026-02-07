from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
#from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3

load_dotenv()

#chatbot = graph.compile()

llm = ChatOllama(model= "mistral") 

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn= conn)


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retreive_all_threads():
    all_threads =set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return(list(all_threads))


#for streaming

for message_chunk,metadata in chatbot.stream({"messages": [HumanMessage(content='what is the recipy to make mango juice ')]},
               config= {"configurable": {"thread_id": "thread-3"}} ,
               stream_mode= 'messages'
               ):
    if message_chunk.content:
        print(message_chunk.content, end=" " ,flush=True)

""" CONFIG = {
    "configurable": {
        "thread_id": "thread_1"
    }
} """


""" response = chatbot.invoke(
                {"messages": [HumanMessage(content='hi my name is shubham')]},
                config=CONFIG
            )

print(chatbot.get_state(config= CONFIG).values['messages'])

#print(type(stream)) """

# This code is used for testing the database

"""
CONFIG = {'configurable': {'thread_id':'thread-3'}}

response = chatbot.stream(
                {"messages": [HumanMessage(content= 'hello my name is kulkya ')]},
                cofig= CONFIG           
            )

print(response)"""