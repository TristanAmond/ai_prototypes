from dotenv import load_dotenv
import streamlit as st
import candidate_sim

# Load the .env file
load_dotenv()

# --- BEGIN STREAMLIT ---
# Set the page title and layout
st.set_page_config(page_title="DebateFightNight", layout="wide", page_icon=':us:')
st.title('Debate Fight Night')
container = st.container()
candidates = []
col1, col2 = container.columns([1,2])
with col1:
    cola, colb = col1.columns(2)
    # Set candidates
    with cola:
        candidate1 = st.text_input('Candidate', value="George Washington", placeholder="George Washington", key='candidate1')
        candidates.append(candidate1)
        candidate2 = st.text_input('Candidate', value="Alex Trebek", key='candidate2')
        candidates.append(candidate2)
    with colb:
        candidate3 = st.text_input('Candidate', value="Dr. Zoidberg from Futurama", key='candidate3')
        candidates.append(candidate3)
        candidate4 = st.text_input('Candidate', value="Snoopy", key='candidate4')
        candidates.append(candidate4)
    # Set topic
    candidate_topic = st.text_input('Debate Issue', value="Should the US national food be ice cream? If so, "
                                                                "what flavor?", key='issue')
    colz, coly = st.columns(2)
    with colz:
        # Set number of rounds
        round_count = st.number_input('Number of rounds', min_value=1, max_value=10, value=5, step=1)
    with coly:
        # Set the button
        begin_button = st.button('Begin debate',
                             use_container_width=True)

    moderator_output = st.markdown("")

with col2:
    cand1, cand2, cand3, cand4 = col2.columns(4)
    candidate_images = []
    with cand1:
        st.session_state.candidate_image1 = st.empty()
        candidate_images.append(st.session_state.candidate_image1)
    with cand2:
        st.session_state.candidate_image2 = st.empty()
        candidate_images.append(st.session_state.candidate_image2)
    with cand3:
        st.session_state.candidate_image3 = st.empty()
        candidate_images.append(st.session_state.candidate_image3)
    with cand4:
        st.session_state.candidate_image4 = st.empty()
        candidate_images.append(st.session_state.candidate_image4)
    st.divider()
    debate_output = st.markdown("")

if begin_button:
    with st.spinner("Generating..."):
        for item in candidates:
            generated_image = candidate_sim.generate_character_image(item)

            try:
                current_candidate_image = candidate_images.pop(0)
                current_candidate_image = current_candidate_image.image(generated_image)
            except IndexError:
                break

        candidate_sim.st_mod(debate_output,
                              moderator_output,
                              candidates,
                              candidate_topic,
                              round_count,
                              st.session_state)
