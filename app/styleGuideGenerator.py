from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import re
import json
import os

load_dotenv(find_dotenv())
client = OpenAI()

def gen_style_template(name, transcript_path="placeholder", use_existing=True):
    shortened_name = name.replace(" ", "").lower()
    # automate retrieving the correct transcript for a given person
    if transcript_path == "placeholder":
        transcript_path = f"{shortened_name}_transcript.txt"

    # if style_guide for this person already exists AND we purposefully set ignore_existing to True
    if os.path.isfile(f'{shortened_name}_styleguide.json') and use_existing:
        print(f"File f'{shortened_name}_styleguide.json' already exists.")
        # Load existing, convert to string, and return
        with open(f'{shortened_name}_styleguide.json', 'r') as file:
            json_data = json.load(file)
        json_string = json.dumps(json_data)
        return json_string

    # Define the Style template example
    system_template = f"""### Style Template Example
    {{"speaker": "Donald Trump", "lexical_choice": [ "tremendous", "believe me", "very much" ], "sentence_structure": "Fragmented and simple declarative sentences", "rhetorical_techniques": [ "Direct address ('my friends', 'folks')", "Rhetorical questions", "Hyperbole" ], "engagement_style": "Direct, confrontational, involves audience with questions and commands", "emotional_tone": "Defiant, assertive, aggressive", "thematic_content": [ "national security", "business competency", "media criticism" ], "examples_from_transcripts": [ "These are bad people.", "They’re destroying our country.", "It was a rigged trial." ], "personal_values": "Strong nationalistic sentiments, pro-business outlook", "avoid_list": ["Use of politically left-leaning rhetoric or liberal policy endorsement", "Submissive or passive language", "Complex techno-scientific jargon", "Expressions of vulnerability or uncertainty", "Detailed, nuance-driven or introspective arguments", "Avoiding direct answers or showing hesitancy", "Low energy or passive engagement styles"], "instructions": "Generate responses that employ the above elements faithfully, synthesizing them into coherent outputs that maintain the stylistic and emotional consistency of the speaker’s public persona."}}
    ### Parent Prompt for Generating Output Prompt Template
    This parent prompt outlines how the GPT-4 template should be filled out based on the provided name and transcripts. It acts essentially as a guide for creating the above output prompt template when new input (a different speaker's name and transcript) is introduced:
    ### TASK: Define a new output prompt template for a future GPT-4 agent to mimic a given speaker's style: **
    1. Use the name of this public figure {{PUBLIC FIGURE NAME}} and the transcripts {{TRANSCRIPTS}} provided in the user message information {{INFORMATION}}.
    2. Analyze multiple speech transcripts of the speaker and complete the output template from ** Style Template Example**:
        - Utilize text-processing techniques to extract frequently used words and phrases for "lexical_choice".
        - Observe and summarize common sentence structures as "sentence_structure".
        - Identify distinctive rhetorical strategies employed by the speaker and list them under "rhetorical_techniques".
        - Note how the speaker engages with the audience and describe this under "engagement_style".
        - Capture the predominant emotional undertone of the speaker’s public messages in "emotional_tone".
        - List 15-20 central themes or topics frequently addressed by the speaker in the transcript under "thematic_content", also leveraging public knowledge about the speaker themselves to add a larger overview.
        - Provide 15-20 direct, exact, word for word quotes from the transcripts that illustrate the speaking style and unique character of the speaker, especially cases that defy average speaking behaviour, adding them to "examples_from_transcripts".
        - Understand and define the speaker’s core personal values and insights reflected in their transcript and public information under "personal_values".
        - Determine and list communication elements that are not characteristic of the speaker by reviewing the transcripts and noting any out-of-character phrases, themes, or styles that are not typical of the speaker. Include these elements in an "avoid_list" on how, based on general knowledge about the speaker and discrepancies observed in the style of the transcripts, certain words, phrases, or behaviors should be avoided to maintain stylistic authenticity. Compare against public records and interviews to validate these choices.
    3. Compose additional step-by-step instructions for the future GPT agent on how to integrate all extracted features to maintain the authenticity of the speaker’s style, and detail them under a supplemental section labeled "instructions". Ensure it directs the GPT agent to weigh its response styles heavily towards the details provided in the “Style Template” creates in 2.
    4. Ensure two things. 1. The style template's JSON format is STRICTLY followed. 2. That there is NEVER any racially intolerant content included in the style guide generated."""

    with open(transcript_path, 'r', encoding='utf-8') as file:
        transcripts = file.read()

    user_prompt= f"""### INFORMATION: Use the following name and transcript of text to complete the task ** 
    PUBLIC FIGURE NAME: {name}
    TRANSCRIPTS: "{transcripts}" 
    """

    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system",
             "content": system_template},
            {"role": "user", "content": user_prompt}
        ]
    )
    print(completion)
    json_data = re.search(r'json\n(\{.*?\})\n', completion.choices[0].message.content, re.DOTALL)
    print(json_data)
    if json_data:
        json_output = json_data.group(1)
        shortened_name = name.replace(" ", "").lower()
        with open(f'{shortened_name}_styleguide.json', 'w') as f:
            f.write(json_output)
    else:
        print("No JSON data found.")
        return None

    return json_output

if __name__ == '__main__':
    name = "Michael Pollan"
    template_string = gen_style_template(name, use_existing=False)
    print(template_string)
