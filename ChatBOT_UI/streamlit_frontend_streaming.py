import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

#a chat bot which will mimic the user chat 
#but its not able to store these messages
#st.session_state -> dictionary in which the input wont erase whenever you press enter , 
                    # the valueswill be resetted only when you manually reset it through page refresh

# st.session_state -> dict -> 
CONFIG = {"configurable": {"thread_id": "thread-1"}}

st.title("🤖 LangGraph Chatbot")

# ---------------- Session State ----------------
if "message_history" not in st.session_state:
    st.session_state.message_history = []

# ---------------- Display Chat History ----------------
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- Input ----------------
user_input = st.chat_input("Type here...")

if user_input:
    # store user message in UI memory
    st.session_state.message_history.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # IMPORTANT: send ONLY new message (LangGraph handles history)
    #response = chatbot.invoke( {"messages": [HumanMessage(content=user_input)]}, config=CONFIG )

    #ai_message = response["messages"][-1].content

    # store AI message in UI memory
    #st.session_state.message_history.append({"role": "assistant", "content": ai_message})

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk,metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
               config= {"configurable": {"thread_id": "thread-1"}} ,
               stream_mode= 'messages'
            )
        )
    st.session_state.message_history.append({"role": "assistant", "content": ai_message})    
    #    st.markdown(ai_message)