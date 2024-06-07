# masterclassConversational


## Installation

Install the LangChain CLI if you haven't yet

```bash
pip install -U langchain-cli
```

## STAGE A: Corpus generalization
Input:
- File path to audio file (TODO: take a folder of audio files instead)
- Name of the speaker

Flow:
- Transcribe the audio file(s) with Whisper ASR
- Process the transcriptions into split text segments with HDP Topic Modeling guided text segmentation
- Use the OpenAI embed-text-03-large embedding model to create embeddings for each text segment
- Index the embeddings into a Pinecone Vector DB
- Using the GPT4 engine, generate a master styling template guide from the transcripts provided.

DEPR:Generate a broad corpus analysis of General Style Features across the language corpus

Used Scripts:
- audio_transcription.py: Transcribe the audio file(s) with Whisper ASR and save the transcription to a text file
- textSplitHDPTopic.py: Splits, embeds, and indexes into Pinecone the transcription. Creates the text segments with HDP Topic Modeling guided text segmentation, and OpenAI embedding.
- styleGuideGenerator: Generates a master styling template guide utilizing GPT4-turbo, the transcripts provided, and the name of the speaker

## STAGE B: User Engagement with Streamlit
Input:
- User query (text)

Flow:
- User Query is embedded using the OpenAI embed-text-03-large model, then, in parallel we kick off:
- FLOW 1, Step 1: We use the embedding to query the Pinecone Vector DB for quantitatively similar text segments (related knowledge)
- FLOW 1, Step 2: We then build the related knowledge into a coherent knowledge supplement to the GPT4 model prompt to aid/focus its response
- FLOW 2, Step 1: We use the embedding to query the Pinecone Vector DB for topic similar stylistic features
- FLOW 2, Step 2: We then build the stylistic features into a coherent style guide, merged with  our broad corpus analysis of general style from Stage A to aid + constrain the GPT 4 in speaking stylistically similar
- We merge Flow 1 and Flow 2 results into a coherent GPT4 model prompt, and generate a response from the model
- (OPTIONAL) We verify stylistic features and knowledge content was respected in the model response
- We return the model response to the user

Used Scripts:
- responseAgentGenerate.py: Generates a response from the GPT4 model using the master styling guide from Stage A, the and the knowledge query results from our RAG chain



## Setup LangSmith (Optional)
LangSmith will help us trace, monitor and debug LangChain applications. 
You can sign up for LangSmith [here](https://smith.langchain.com/).

```shell
docker build . -t my-langserve-app
```

## TODO / Improvements
- Update README with new architecture.
- Significantly more mature RAG methods for similarity improvements and hallucination reduction a
- Better tracing and explainability
- SHAP evaluation methods for Precision and Recall
- Implementing TTS driven Voice modelling of AI outputs with SVC/RVC/internal models
- Direct feedback implementation
- Implement Chat history retention and summarization to aid continuous conversation knowledge retention for better answers
- Distributed deployment
- Faster embedding
- More efficient vector storage, upgrading compute