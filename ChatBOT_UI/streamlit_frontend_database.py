import streamlit as st
from langgraph_backend_database import chatbot,retreive_all_threads
from langchain_core.messages import HumanMessage, BaseMessage
import uuid

st.set_page_config(page_title="LangGraph Chatbot", layout="centered")

# ---------------- Utility ----------------
def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def reset_chat():
    new_thread = generate_thread_id()
    st.session_state["thread_id"] = new_thread
    add_thread(new_thread)
    st.session_state["message_history"] = []
    st.rerun()

def load_conversation(tid):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": tid}}
    )
    return state.values.get("messages", [])

# ---------------- Session State INIT ----------------
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retreive_all_threads()

add_thread(st.session_state["thread_id"])

# ---------------- LangGraph CONFIG ----------------

#CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}
CONFIG = {"configurable": {'thread_id': st.session_state['thread_id']},
          'metadata': {
              'thread_id': st.session_state['thread_id']
          },
          'run_name': "chat_turn",
}

# ---------------- Sidebar ----------------
st.sidebar.title("LangGraph Chatbot")
st.sidebar.header("My Conversations")

for tid in st.session_state["chat_threads"]:
    if st.sidebar.button(tid):
        st.session_state["thread_id"] = tid

        messages = load_conversation(tid)
        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"

            temp_messages.append(
                {"role": role, "content": msg.content}
            )

        st.session_state["message_history"] = temp_messages
        st.rerun()

if st.sidebar.button("➕ New Chat"):
    reset_chat()

st.sidebar.header("Current Thread ID")
st.sidebar.code(st.session_state["thread_id"])

# ---------------- Main UI ----------------
st.title("🤖 LangGraph Chatbot")

# ---------------- Display History ----------------
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- Input ----------------
user_input = st.chat_input("Type here...")

if user_input:
    # User message
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # AI streaming
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            chunk.content
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config={
                    "configurable": {
                        "thread_id": st.session_state["thread_id"]
                    }
                },
                stream_mode="messages",
            )
        )

    # Save AI message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
