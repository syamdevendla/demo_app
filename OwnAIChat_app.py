import streamlit as st
import own_config
import speech_text
import shutil
import tempfile
import os
from audiorecorder import audiorecorder
import requests
import validators

# from translate import Translator

from convert_to_text import TokenizerAndAgent

st.set_page_config("Chat with OWN Data")
st.subheader('''
**Search images/pdf by asking question through (text/voice)**
****Your data is secured[option to delete uploaded data]**** ''')

st.title("Own GenAI Chat🔥")

QUESTION_HISTORY: str = 'question_history'
USER_QUESTION: str = 'user_question'
AI_RESPONSE: str = 'ai_response'
AUDIO_INPUT: audiorecorder = ""
AUDIO_EVENT_TRIGGERED: bool = True
RELOAD_AGENT_REQUIRED = False
USER_SELECTED_LANGUAGE: str = "english"

list_of_files_uploaded = []

if RELOAD_AGENT_REQUIRED not in st.session_state:
    st.session_state[RELOAD_AGENT_REQUIRED] = False

if QUESTION_HISTORY not in st.session_state:
    st.session_state[QUESTION_HISTORY] = []

if USER_QUESTION not in st.session_state:
    st.session_state[USER_QUESTION] = ""

if AI_RESPONSE not in st.session_state:
    st.session_state[AI_RESPONSE] = ""

if AUDIO_INPUT not in st.session_state:
    st.session_state[AUDIO_INPUT] = ""

if AUDIO_EVENT_TRIGGERED not in st.session_state:
    st.session_state[AUDIO_EVENT_TRIGGERED] = True

if USER_SELECTED_LANGUAGE not in st.session_state:
    st.session_state[USER_SELECTED_LANGUAGE] = "english"


@st.cache_resource()
def prepare_agent() -> TokenizerAndAgent:
    return TokenizerAndAgent()


def submit():
    st.session_state[USER_QUESTION] = st.session_state.query
    st.session_state.query = ''
    st.session_state[AUDIO_EVENT_TRIGGERED] = False


def on_change_select_cb():
    print("on_change_select_cb entered")
    st.session_state[AUDIO_EVENT_TRIGGERED] = False


def intro_text():
    with st.sidebar.expander("Click to see application info:"):
        st.write(f""" Ask questions about:
- Personal content, will search in the uploaded documents
- Latest events
- Wikipedia Content
- multi language support, voice based
- Its secured, option to delete all uploaded docs.
- Voice based search enabled and response could be played through audio.
- ***Please note: its still work in progress***
    """)


intro_text()


def process_uploaded_files(uploaded_files):
    for file in uploaded_files:
        if file is not None:
            with tempfile.NamedTemporaryFile(dir="uploaded_data/", delete=False) as f:
                f.write(file.getbuffer())
                temp = f.name
                destination_source = "uploaded_data/" + file.name
                destination_backup = "data_backup/" + file.name
                shutil.copyfile(temp, destination_source)
                shutil.copyfile(temp, destination_backup)
                f.close()
                os.unlink(f.name)


with st.sidebar.form("my-upload-form", clear_on_submit=True):
    uploaded_files = st.file_uploader("For personal content, upload related Docs", type=["pdf", "jpg", "jpeg", "png"],
                                      accept_multiple_files=True)
    submit_uploaded_files = st.form_submit_button("upload")

if submit_uploaded_files and uploaded_files is not None:
    st.session_state[AUDIO_EVENT_TRIGGERED] = False
    process_uploaded_files(uploaded_files)
    prepare_agent().reload_module()
    # if list_of_files_uploaded:
    # st.session_state[RELOAD_AGENT_REQUIRED] = True


def init_stream_lit():
    # if st.session_state[RELOAD_AGENT_REQUIRED]:
    # st.cache_resource.clear()
    # st.session_state[RELOAD_AGENT_REQUIRED] = False

    # agent_executor: AgentExecutor = prepare_agent()

    input_question_translated_to_eng = ''

    simple_chat_tab, historical_tab = st.tabs([":blue[***AI Chat***]", ":black[***Session Chat History***]"])
    with simple_chat_tab:

        st.text_input(":red[Your question ❓]", key='query', on_change=submit)
        col1, col2 = st.columns(2)
        with col1:
            st.session_state[AUDIO_INPUT] = audiorecorder("🎙️ speak", "🎙️ stop")
        with col2:
            selected_lang_option = st.selectbox(
                "select speech language",
                ("english", "hindi", "telugu", "arabic"),
                on_change=on_change_select_cb,
            )
            st.session_state[USER_SELECTED_LANGUAGE] = selected_lang_option
            print("selected_lang_option: ", selected_lang_option)

        if not st.session_state[AUDIO_EVENT_TRIGGERED]:
            st.session_state[AUDIO_INPUT] = ""
        st.session_state[AUDIO_EVENT_TRIGGERED] = True

        if len(st.session_state[AUDIO_INPUT]) and not st.session_state[USER_QUESTION]:
            st.session_state[USER_QUESTION] = speech_text.audio_to_text_Convertion(
                st.session_state[AUDIO_INPUT].export("audio.wav", format="wav"),
                st.session_state[USER_SELECTED_LANGUAGE])

            if os.path.isfile("audio.wav"):
                os.remove("audio.wav")

        st.session_state[AUDIO_INPUT] = ""

        if st.session_state[USER_QUESTION]:
            with st.spinner('Please wait ...'):
                try:
                    question_placeholder = st.empty()
                    player_placeholder = st.empty()
                    res_placeholder = st.empty()
                    query_output = {}
                    if st.session_state[USER_SELECTED_LANGUAGE] != "english":
                        input_question_translated_to_eng = speech_text.translate_to_english(
                            st.session_state[USER_QUESTION], st.session_state[USER_SELECTED_LANGUAGE])

                    else:
                        input_question_translated_to_eng = st.session_state[USER_QUESTION]

                    if st.session_state[USER_QUESTION] == "Could not understand your audio, PLease try again !":
                        res_placeholder.write("🔥 :green[Own-AI : ]" f":green[{st.session_state[USER_QUESTION]}]")
                        st.session_state[USER_QUESTION] = ""
                    else:
                        question_placeholder.write(f":red[Q: {st.session_state[USER_QUESTION]}]")
                        # query = "whats the website mentioned to Get a quote for the completion of  template "
                        query_output = prepare_agent().execute_query(input_question_translated_to_eng, 1)
                        st.session_state[AI_RESPONSE] = query_output["answer"]
                        if not st.session_state[AI_RESPONSE]:
                            st.session_state[AI_RESPONSE] = "No response, possible reason: server down"

                        if st.session_state[AI_RESPONSE] and st.session_state[USER_SELECTED_LANGUAGE] != 'english':
                            output_translations = speech_text.translate_eng_to_selected_lang(
                                st.session_state[AI_RESPONSE], st.session_state[USER_SELECTED_LANGUAGE])
                            st.session_state[AI_RESPONSE] = output_translations

                        res_placeholder.write("🔥 :green[Own-AI : ]" f":green[{st.session_state[AI_RESPONSE]}]")
                    if st.session_state[AI_RESPONSE] and st.session_state[USER_QUESTION]:
                        audio_out_file = speech_text.output_text_to_speak(st.session_state[AI_RESPONSE],
                                                                          st.session_state[USER_SELECTED_LANGUAGE])
                        player_placeholder.audio(audio_out_file)
                        os.remove(audio_out_file)
                        # Display images
                        cols = st.columns((1, 1))
                        image_paths = query_output["page_img"]

                        print("image_paths count: ", len(image_paths))
                        for i, image_path in enumerate(image_paths):
                            # image_path = "data_backup/".join(images)
                            if os.path.isfile(image_path):
                                cols[i % 2].image(image_path, clamp=True)

                        page_nums = query_output["page_num"]

                        st.session_state[QUESTION_HISTORY].append(
                            (st.session_state[USER_QUESTION], st.session_state[AI_RESPONSE]))
                    st.session_state[USER_QUESTION] = ""
                    st.session_state[AI_RESPONSE] = ""

                except Exception as e:
                    st.error(f"Error occurred: {e}")

    with historical_tab:
        print("Entered : historical_tab")
        for q in st.session_state[QUESTION_HISTORY]:
            question = q[0]
            if len(question) > 0 and question not in st.session_state[QUESTION_HISTORY]:
                st.write(f":red[Q: {question}]")
                st.write(f":green[A: {q[1]}]")


if __name__ == "__main__":
    init_stream_lit()
