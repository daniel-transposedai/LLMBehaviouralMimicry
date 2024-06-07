import sys
sys.path.append('../../')
import streamlit as st

from masterclassConversational.app.relevantContentQuery import get_vectorstore, create_query_chain, format_vectorcontext

from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv, find_dotenv




def get_response(user_input):
    """Get the response for the given user query.

    This function retrieves the response for the user query
    using the conversational RAG chain.

    Args:
        user_input (str): The user query.

    Returns:
        str: The response to the user query.
    """


    rag_chain = create_query_chain(st.session_state["styleguide_selection"], st.session_state.vector_store)

    response = rag_chain.invoke(user_input)

    # return textwrap.fill(response, width=70)
    return f"***{get_first_name(st.session_state['current_speaker'])}***: " + response

def get_first_name(name):
    name_without_asterisks = name.replace('*', '')
    name_parts = name_without_asterisks.split(' ')
    first_name = name_parts[0]
    return first_name

def reset_conversation():
    st.session_state.conversation = []
    st.session_state.chat_history = []
    st.session_state['styleguide'] = None


def main():

    load_dotenv(find_dotenv())
    # App Config
    st.set_page_config(page_title="Learn with Masterclass Instructors", page_icon="🦜")
    st.title("_Learn_ with :blue[MasterClass Instructors] 🤖")

    # App Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown(
            """
            This chatbot interfaces with a
            [LangChain](https://python.langchain.com/docs/get_started/introduction)
            agent designed to help you learn alongside your favourite celebrities!
            """
        )

        st.header(":red[Settings]", divider='rainbow')

        with st.form(key="my_form"):
            genre = st.radio(
                "Who do you want to talk to!",
                ["***Kevin Hart***", "***Michael Pollan***"],
                captions=["Laugh out loud with the best!", "Learn to eat well!"])

            # Every form must have a submit button.
            submitted = st.form_submit_button("Start/Reset my chat!", on_click=reset_conversation)
            if submitted:
                # Well take the value of genre here and clear the chat and reset
                st.session_state.vector_store = get_vectorstore(format_vectorcontext(genre))
                st.session_state['styleguide_selection'] = format_vectorcontext(genre)
                st.session_state['current_speaker'] = genre
                st.session_state.chat_history = [
                    AIMessage(
                        content=f"***{get_first_name(st.session_state['current_speaker'])}***: Hey there, I'm *{st.session_state['current_speaker']}*. Let me help you learn with my unique style!")
                ]

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # For persistent variables : Session State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Select who you want to speak to on the left panel first!")
            ]

    # Persistent Vector Store
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore(format_vectorcontext(genre))

    # User Input
    user_query = st.chat_input("Type your message ✍")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Conversation
    if st.session_state.chat_history is not None:
        for message in st.session_state.chat_history:
            MESSAGE_TYPE = "AI" if isinstance(message, AIMessage) else "Human"
            with st.chat_message(MESSAGE_TYPE):
                st.write(message.content)

if __name__ == '__main__':
    main()