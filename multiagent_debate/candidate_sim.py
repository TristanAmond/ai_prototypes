import os
from typing import List

import numpy as np
import openai
import tenacity
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

import simulations
from simulations import (
    DialogueAgent,
    DialogueSimulator,
    BiddingDialogueAgent,
)

# --- CONSTANTS SETUP ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
default_candidate_image = "https://instagram.fphl1-1.fna.fbcdn.net/v/t51.2885-15/45480315_765063740512941_6686185249891644973_n.jpg?stp=dst-jpg_e35&_nc_ht=instagram.fphl1-1.fna.fbcdn.net&_nc_cat=109&_nc_ohc=HXO4eCZkWEMAX9tBt-k&edm=ACWDqb8BAAAA&ccb=7-5&ig_cache_key=MTkyNDE2MzM3NjAxMjExNDkzOQ%3D%3D.2-ccb7-5&oh=00_AfC07bMyplVEhb0uY-qGpSnSi9SMMqXA4RFafK4dGrxhAw&oe=64557846&_nc_sid=1527a3"

player_descriptor_system_message = SystemMessage(
    content="You can add detail to the description of each candidate.")

word_limit = 50

bid_parser = simulations.BidOutputParser(
        regex=r'<(\d+)>',
        output_keys=['bid'],
        default_output_key='bid')


# --- FUNCTIONS ---
def generate_game_description(character_names, topic):
    game_description = f"""Here is the topic for the debate: {topic}.
        The presidential candidates are: {', '.join(character_names)}."""
    return game_description


def generate_character_description(character_name, game_description):
    character_specifier_prompt = [
        player_descriptor_system_message,
        HumanMessage(content=
                     f"""{game_description}
            Please reply with a creative description of the candidate, {character_name}, in {word_limit}
            words or less, that emphasizes their personalities. Speak directly to {character_name}.
            Do not add anything else."""
                     )
    ]
    character_description = ChatOpenAI(temperature=1.0)(character_specifier_prompt).content
    return character_description


def generate_character_header(game_description, character_name, character_description, debate_topic):
    return f"""{game_description}
Your name is {character_name}.
You are a debate candidate.
Your description is as follows: {character_description}
You are debating the topic: {debate_topic}.
Your goal is to be as creative as possible and make the voters think you are the best candidate.
"""


def generate_character_system_message(character_name, character_header, debate_topic):
    return SystemMessage(content=(
        f"""{character_header}
You will speak in the style of {character_name}, and exaggerate their personality.
You will come up with creative ideas related to {debate_topic}.
Do not say the same things over and over again.
Speak in the first person from the perspective of {character_name}
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of anyone else.
Speak only from the perspective of {character_name}.
Stop speaking the moment you finish speaking from your perspective.
Never forget to keep your response to {word_limit} words!
Do not add anything else.
    """
    ))


def initialize_characters(character_names, debate_topic, game_description):
    character_descriptions = [generate_character_description(character_name, game_description) for character_name in character_names]
    character_headers = [generate_character_header(game_description, character_name, character_description, debate_topic) for
                         character_name, character_description in zip(character_names, character_descriptions)]
    character_system_messages = [generate_character_system_message(character_name, character_headers, debate_topic) for
                                 character_name, character_headers in zip(character_names, character_headers)]

    return character_descriptions, character_headers, character_system_messages


def generate_character_bidding_template(character_headers):
    character_bidding_templates = []
    for character_header in character_headers:
        bidding_template = (
                f"""{character_header}
        
        ```
        {{message_history}}
        ```
        
        On the scale of 1 to 10, where 1 is not contradictory and 10 is extremely contradictory, rate how 
        contradictory the following message is to your ideas.
        
        ```
        {{recent_message}}
        ```
        
        {bid_parser.get_format_instructions()}
        Do nothing else.
            """)
        character_bidding_templates.append(bidding_template)
    return character_bidding_templates


def topic_specifier(debate_topic, game_description, word_limit, character_names):
    topic_specifier_prompt = [
        SystemMessage(content="You can make a task more specific."),
        HumanMessage(content=
                     f"""{game_description}
    
            You are the debate moderator.
            Please make the debate topic more specific. 
            Frame the debate topic as a problem to be solved.
            Be creative and imaginative.
            Please reply with the specified topic in {word_limit} words or less. 
            Speak directly to the candidates: {*character_names,}.
            Do not add anything else."""
                     )
    ]
    specified_topic = ChatOpenAI(temperature=1.0)(topic_specifier_prompt).content

    # print(f"Original topic:\n{debate_topic}\n")
    # print(f"Detailed topic:\n{specified_topic}\n")
    return specified_topic


# --- BID HANDLER FUNCTIONS ---
@tenacity.retry(stop=tenacity.stop_after_attempt(2),
                wait=tenacity.wait_none(),  # No waiting time between retries
                retry=tenacity.retry_if_exception_type(ValueError),
                before_sleep=lambda retry_state: print(
                    f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."),
                retry_error_callback=lambda retry_state: 0)  # Default value when all retries are exhausted
def ask_for_bid(agent):
    """
    Ask for agent bid and parses the bid into the correct format.
    """
    bid_string = agent.bid()
    bid = int(bid_parser.parse(bid_string)['bid'])
    return bid


def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    bids = []
    for agent in agents:
        bid = ask_for_bid(agent)
        bids.append(bid)

    # randomly select among multiple agents with the same bid
    max_value = np.max(bids)
    max_indices = np.where(bids == max_value)[0]
    idx = np.random.choice(max_indices)

    print('Bids:')
    for i, (bid, agent) in enumerate(zip(bids, agents)):
        print(f'\t{agent.name} bid: {bid}')
        if i == idx:
            selected_name = agent.name
    print(f'Selected: {selected_name}')
    print('\n')
    return idx


def generate_character_image(name, img_size="256x256"):
    try:
        print(f"Generating image for {name}")
        prompt = f"a drawing of {name} in courtroom sketch style"
        model = "image-alpha-001"
        response = openai.Image.create(
            prompt=prompt,
            model=model,
            size=img_size
        )
        image_url = response['data'][0]['url']
        print(f"{name} URL: {image_url}")
    except Exception as e:
        print(e)
        return default_candidate_image

    if ".png"  in image_url:
        return image_url
    else:
        return default_candidate_image


# Write the message to the appropriate output component based on the name
def write_to_component(name, message, messages, colors, output_component, moderator_component):
    color = colors.get(name)
    if name == "Debate Moderator":
        moderator_component.markdown(f'**:gray[MODERATOR]**\n{message}', unsafe_allow_html=True)
    else:
        messages.append(f"**:{color}[{name}]**: {message}")
        output_component.markdown('\n\n'.join(reversed(messages)), unsafe_allow_html=True)


def write_to_image(name, character_images, session_state):
    session_state['candidate_image'].image(character_images.get(name))


# Main Game loop
def st_mod(output_component, moderator_component, character_names, debate_topic, rounds, session_state):
    with get_openai_callback() as cb:
        # generate game description, characters, and character bidding templates
        game_description = generate_game_description(character_names, debate_topic)
        character_descriptions, character_headers, character_system_messages = initialize_characters(
            character_names, debate_topic, game_description
        )
        character_bidding_templates = generate_character_bidding_template(character_headers)

        # Generate Characters as BiddingDialogueAgents
        characters = []
        for character_name, character_system_message, bidding_template in zip(character_names, character_system_messages,
                                                                              character_bidding_templates):
            characters.append(BiddingDialogueAgent(
                name=character_name,
                system_message=character_system_message,
                model=ChatOpenAI(temperature=0.2),
                bidding_template=bidding_template,
            ))
    print("Token Count callback (No Trim): ", cb)

    # Define round and message history
    max_iters = rounds
    n = 0
    messages = []
    character_colors = {character_names[0]: 'red',
                        character_names[1]: 'blue',
                        character_names[2]: 'green',
                        character_names[3]: 'yellow'}

    # Initialize simulator
    simulator = DialogueSimulator(
        agents=characters,
        selection_function=select_next_speaker
    )
    # Specify the topic
    specified_topic = topic_specifier(debate_topic, game_description, word_limit, character_names)
    simulator.reset('Debate Moderator', specified_topic)
    write_to_component('Debate Moderator', specified_topic, messages, character_colors,
                       output_component, moderator_component)
    print(f"(Debate Moderator): {specified_topic}")
    print('\n')

    while n < max_iters:
        name, message = simulator.step()
        write_to_component(name, message, messages, character_colors, output_component, moderator_component)
        print(f"({name}): {message}")
        print('\n')
        n += 1
