from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from pinecone import Pinecone
import os
import textwrap
import pinecone
import json
from dotenv import load_dotenv, find_dotenv


# get API Keys
load_dotenv(find_dotenv())
#os.environ["LANGCHAIN_TRACING_V2"] = "true"

def format_vectorcontext(name):
    """Set the OpenAI API key in the environment variable.
    Args:
    key (str): The OpenAI API key to set.
    """
    clean_text = name.replace('*', '').replace(' ', '').lower()
    print(clean_text)

    return clean_text


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_vectorstore(name):
    query_embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    # Init the PineconeVectorStore for the index
    vectorstore = PineconeVectorStore(index_name=name, embedding=query_embedding, namespace="masterclass")

    return vectorstore


def retrieve_styleguide(name):
    # Load the styleguide from the json file
    shortened_name = name.replace(" ", "").lower()
    with open(f'{shortened_name}_styleguide.json', 'r') as file:
        json_data = json.load(file)
    json_string = json.dumps(json_data)
    return json_string

def create_query_chain(styleguide_selection, vectorstore):
    # Initialize the OpenAIEmbeddings class with the text-embedding-3-large model (need to keep dimensionality equal to index)
    query_embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    # Init the PineconeVectorStore for the index
    #vectorstore = PineconeVectorStore(index_name=index_name, embedding=query_embedding, namespace="testing")
    retriever = vectorstore.as_retriever(k = 10)

    llm = ChatOpenAI(model="gpt-4o")

    speaker_styleguide = retrieve_styleguide(styleguide_selection)


    # Removing {} from the json styleguide as it screws with the Langchain formatting. Janky, but it works
    filtered_styleguide = speaker_styleguide.replace('{', '[').replace('}', ']')
    system_prompt = (
            "## **Guidance Prompt for GPT-4 Agent Using Style Template**"
            "As you prepare to respond, ensure that you strictly adhere to the style template provided. Your responses should entirely embody the linguistic characteristics, emotional tone, and thematic content specific to the outlined persona. Utilize the lexical choices, sentence structures, rhetorical techniques, styles and areas to avoid, and other traits as detailed in the template, crafting your replies to reflect the unique speaking and engagement style of the persona. You are not permitted to deviate from this persona in any response, regardless of context. Prioritize consistency with the speaker's known public persona in all aspects of your interaction. Prioritize using the following pieces of retrieved context from the 'Knowledge Context' section to answer the users query as they are primary sources of information. You may use your overall knowledge of the speaker to aid in the knowledge component of the question, however you must bias your answer heavily towards the knowledge provided. If you do not know the answer, you should politely declare that you do not know the answer. In short, integrate all elements of the style template and Knowledge Context to synthesize responses that maintain the stylistic, emotional, and factual authenticity of the speaker’s public demeanor and knowledge."
            "/n/n "
            "### Style Template" +
            filtered_styleguide +
            "/n/n "
            "Knowledge Context: {context}"
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "User Query: {question}"),
        ]
    )


    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
    )

    return rag_chain

if __name__ == "__main__":
    # Test the function
    from styleGuideGenerator import gen_style_template
    #gen_style_template("Michael Pollan", "transcription.txt")
    #speaker_styleguide = gen_style_template(speaker_name)

    speaker_name = "Michael Pollan"
    query1 = "What is the saying regarding white bread?"
    styleguide_selection = format_vectorcontext(speaker_name)

    rag_chain = create_query_chain(styleguide_selection, get_vectorstore(styleguide_selection))
    response = rag_chain.invoke(query1)
    print(textwrap.fill(response, width=70))







