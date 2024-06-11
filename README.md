# masterclassConversational

# Quick Start 
## Installation

Install the LangChain CLI if you haven't yet

```bash
pip install -U langchain-cli
```

And the required packages

```bash
pip install -r requirements.txt
```

## Setup Environment Variables
Create a ```.env``` file with the following variables:

OPENAI_API_KEY="Gather from [here](https://platform.openai.com/api-keys)"
PINECONE_API_KEY="Gather from 'API Keys' in the [Pinecone console](https://app.pinecone.io/organizations/-/projects)"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="Gather from [here](https://smith.langchain.com/settings)"


## Setup LangSmith 
LangSmith will help us trace, monitor and debug LangChain applications. 
You can sign up for LangSmith [here](https://smith.langchain.com/).


## Deploy Locally
```shell
streamlit run app.py
```

# Process Flow

## STAGE A: Corpus generalization
Input:
- File path to audio file (TODO: take a folder of audio files instead)
- Name of the speaker

Flow:
- Step 1: Transcribe the audio file(s) with Whisper ASR
- Step 2: Process the transcriptions into split text segments with HDP Topic Modeling guided text segmentation
- Step 3: Use the OpenAI ```embed-text-03-large``` embedding model to create embeddings for each text segment
- Step 4: Index the embeddings into a Pinecone Vector DB associated to the particular speaker and masterclass namespace.
- Step 5: Using the GPT4 engine, generate and store locally a master styling template guide from the transcripts provided.

Used Scripts:
- audio_transcription.py: Transcribe the audio file(s) with Whisper ASR and save the transcription to a text file
- textSplitHDPTopic.py: Splits, embeds, and indexes into Pinecone the transcription. Creates the text segments with HDP Topic Modeling guided text segmentation, and OpenAI embedding.
- styleGuideGenerator: Generates a master styling template guide utilizing GPT4-turbo, the transcripts provided, and the name of the speaker
- *initializePersona.py* (Status: DEVELOPMENT): Command line utility to execute Stage A in one broad stroke. 
- *styleFeatureDetermination.py* (DEPRECATED, Status: V2 in DEVELOPMENT): Generates a broad corpus analysis of Linguistic Style Features across the reference corpus. 

## STAGE B: User Engagement with Streamlit
Input:
- User query (text)

Flow:
- Step 1: User Query is embedded using the OpenAI ```embed-text-03-large model``` then, in parallel, we kick off:
- Step 2a, FLOW 1: Use the embedding to query the Pinecone Vector DB for quantitatively similar text segments (related knowledge)
- Step 2b, FLOW 1: Build the related knowledge into a coherent knowledge supplement to the GPT4 model prompt to aid/focus its response
- Step 2a, FLOW 2: Retrieve the relevant style guide related to the current speaker 
- Step 2b, FLOW 2 (Status: DEVELOPMENT): Using the user query as a guide, distill the previous chat history to retrieve a conversational context summary (using either LLM or deterministic techniques)
- Step 3: Merge Flow 1 and Flow 2 results into a coherent GPT4 model prompt, constraining it with our style guide and retrieved knowledge to aid the GPT 4 in responding stylistically similar and informationally relevant manner
- Step 4 - (V2 - Status: DEVELOPMENT): Verify stylistic features, knowledge content, and profanity/vulgarity limits were respected in the model response
- Step 5: Return the model response to the user through the Streamlit UI. 

Used Scripts:
- responseAgentGenerate.py: Generates a response from the GPT4 model using the master styling guide from Stage A, the and the knowledge query results from our RAG chain
- app.py: Streamlit front-end, user RAG chain query handling, and session management. 


# Roadmap & Feature Discussion



<img src="https://user-images.githubusercontent.com/46083287/209074776-f4bf44c2-7bf1-43ee-b4e8-16a7022603da.png" width="80%"></img>

**Figure 1:** Planned Big 5/OCEAN, multi-model SNS architecture - guided from research [here](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00459-1) 

## TODO - Improvements Roadmap
**Conversation Style**

- Implement **chat history retention and summarization** to aid continuous conversation knowledge retention for better answers
- Add direct quotation of related RAG indexes from embedded transcript based on relevancy scoring, emphasizing areas of importance
    - *This naturally happens today, with the Model even modifying output text to highlight content with bolding and italics to emphasize their importance. Let’s promote this behaviour further.*

**Behavioural Processing**

- Enhance style guide and generation template to create weightings across **Big 5/OCEAN personality scoring categories** to further bias output with numeric weighting systems, then test outputs for adherence
    - Use multi-model SNS techniques for social media personality analysis to achieve Big 5/OCEAN scoring, see **Figure 1**.  
    - Create additional, non-Big 5/OCEAN characteristic features with numeric weights (vulgarity, politeness, etc) to bias output generation
- Implement an enhanced pipeline step version of *styleFeatureDetermination.py* to aid in detecting and advising behavioural characteristic elements

**Text Embedding and Vector Processing**

- Optimize embedding and chunking with metadata labeling segments by topic, aiming to characterize text into personal anecdotes, informational remarks, speaking context clues, stories, etc.
    - Tune HDP parameter generation (specifically # of topic clustering and chunking size selection), possibly leveraging alternative non-parametric Bayesian clustering models to set/verify adequate context retention in variable document lengths. As inter-corpus bundles of documents may vary in context length, adaptive measurements for clustering need to be employed here.
- Faster embedding techniques and more efficient clustering
    - Parallel embedding of documents.
    - More efficient vector storage, upgrading compute to reduce P99 from ~10s to a more acceptable range ( ~6-8s)
- Use TruLens or Ragas to implement output evaluation: 
  - RAG Triad-based evaluation for Answer Relevance, and Context Relevance scoring.
  - Precision recall accuracy eval. to verify context retrieval completeness
  - Use Ragas to generate synthetic test questions -> for ground truth + context scoring
  - ROUGE/BLEU for context to output relevancy and summarization accuracy overlap measurement
  - PR/ROC-AUC as core metrics for overall model generation accuracy (if annotationed question, ground truth, answer sets are created)


**Architecture and Platform**

- Significantly more mature RAG lookup methods for top k document similarity improvements and hallucination reduction
- Better tracing and explainability
- Automatic generation of text elements for speakers available for selection
- Implementing TTS driven Voice modelling of AI outputs with SVC/RVC/internal models
- Direct feedback implementation to drive future decision making (in conversation)
- Distributed, multi-user capable deployment (non-streamlit)

**Visual**

- Progress bar
- Custom avatar icons per speaker
- Improved speaker selection menu that will comfortably support multiple speakers
- Suggested initial question options UI elements for quick start
- LLM output text streaming to chat (SPIKE: determine if Streamlit supports it).

**Product “Leapfrogs”**

- **Video Assistant:** Tying this to a real-time Masterclass video stream to drive immediate Q&A capabilities during videos
    - **How:** Tie into media player application to pull current timestamp, then embed course video transcripts with timestamp per word segment to tie into RAG related directly to what the user is currently consuming, or recently consumed, within the video.
    - **Why:** Users of Gen Z & X are consuming content at faster and faster rates, attention and retention are constantly at odds. Ensuring that the user can ask questions about the content at hand and received personalized and expert advice, is critical (***and*** ***cool***)
    - Issues: **Hallucination and Timestamp embedding + recall**
