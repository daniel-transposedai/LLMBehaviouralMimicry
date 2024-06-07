import nltk
from nltk.corpus import stopwords
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
import os
from openai import OpenAI
import logging
import time

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('stopwords')
nltk.download('punkt')

# Setup tokenizer and stopwords
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))


# Function to read and preprocess text from a file
def preprocessText(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    documents = sent_tokenize(text)
    texts = [
        [word for word in tokenizer.tokenize(document) if word not in stop_words]
        for document in documents
    ]
    return texts, documents

# Function to apply Hierarchical Dirichlet Process (HDP) Topic Modelling
def applyHDPTopicModelSegmentation(texts):
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Apply HDP and get dominant topics
    topic_boundaries = getDominantTopic(corpus, dictionary)

    # Segment the text based on topic boundaries
    segmented_texts = []
    start = 0
    for end in sorted(topic_boundaries):
        segmented_texts.append(' '.join(original_sentences[start:end]))
        start = end

    if start < len(original_sentences):
        segmented_texts.append(' '.join(original_sentences[start:]))

    # Example of segments
    for segment in segmented_texts:
        print(segment)

    return segmented_texts


# Function to get embeddings using OpenAI API
def getEmbeddings(texts):
    client = OpenAI()
    embeddings = []
    for text in texts:
        response = client.embeddings.create(input=text,
        model="text-embedding-3-large")
        embeds = [record.embedding for record in response.data]
        embeddings += embeds
    return embeddings


# Function to determine the dominant topic for each sentence
def getDominantTopic(corpus, dictionary):

    # Applying the HDP model
    hdp = models.HdpModel(corpus, id2word=dictionary)

    # Topic Modelling for segmentation creates guide to split based on topic changes and retains context
    topic_boundaries = set()
    current_topic = None
    for i, bow in enumerate(corpus):
        topic_probabilities = hdp[bow]
        dominant_topic = max(topic_probabilities, key=lambda x: x[1])[0] if topic_probabilities else None
        if current_topic != dominant_topic:
            topic_boundaries.add(i)
            current_topic = dominant_topic

    return topic_boundaries


def upsertToVectorDB(embeddings, speaker_name):
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    concatenated_name = speaker_name.replace(" ", "").lower()  # Inputs: Michael Pollan Outputs: michaelpollan
    index_name = f"{concatenated_name}"
    namespace = "masterclass"

    # Create or connect to an existing index
    if index_name not in pc.list_indexes().names():
        # if does not exist, create index
        pc.create_index(
            index_name,
            dimension=len(embeddings[0]),  # dimensionality of text-embed-3-large
            metric='dotproduct',
            spec=spec
        )
        # wait for index to be initialized
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    # connect to index
    index = pc.Index(index_name)
    time.sleep(1)

    model_name = "text-embedding-3-large"
    embeddings = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    logger.log(logging.INFO, "Formatting partitioned text into documents")
    new_docs = [Document(page_content=doc) for doc in segmented_texts]

    logger.log(logging.INFO, "Embedding and indexing documents")
    docsearch = PineconeVectorStore.from_documents(
        documents=new_docs,
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )



if __name__ == '__main__':
    # TESTING
    # Preprocess the text
    file_path = '../michaelpollan_transcript.txt'
    speaker_name= 'michaelpollan'
    texts, original_sentences = preprocessText(file_path)


    # HDP AND TOPIC MODELLING
    segmented_texts = applyHDPTopicModelSegmentation(texts)

    # Get embeddings for each text segment
    embeddings = getEmbeddings(segmented_texts)

    # Upsert to Pinecone
    upsertToVectorDB(embeddings, speaker_name)
    print("Done!")















