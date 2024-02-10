import speech_recognition as sr
from gtts import gTTS  # google text to speech
from googletrans import Translator
import streamlit as st

num = 1


def output_text_to_speak(output, speak_lang='en'):
    global num
    # num to rename every audio file
    # with different name to remove ambiguity
    num += 1
    print("PerSon : ", output)
    req_language = "en"
    # match speak_lang:
    if speak_lang == "english":
        req_language = 'en'
    elif speak_lang == "hindi":
        req_language = 'hi'
    elif speak_lang == "telugu":
        req_language = 'te'
    elif speak_lang == "tamil":
        req_language = 'ta'
    elif speak_lang == "arabic":
        req_language = 'ar'
    else:
        req_language = 'en'

    to_speak = gTTS(text=output, lang=req_language, slow=False)
    # saving the audio file given by google text to speech
    file = "voices_00" + str(num) + ".mp3"
    to_speak.save(file)
    return file


def audio_to_text_Convertion(audiopath, speak_lang='en-IN'):
    req_language = "en-IN"
    # match lang:
    if speak_lang == "english":
        req_language = 'en-IN'
    elif speak_lang == "hindi":
        req_language = 'hi-IN'
    elif speak_lang == "telugu":
        req_language = 'te-IN'
    elif speak_lang == "tamil":
        req_language = 'ta-IN'
    elif speak_lang == "arabic":
        req_language = 'ar-sa'
    else:
        req_language = 'en-IN'
    with sr.AudioFile(audiopath) as source:
        r = sr.Recognizer()
        audio_text = r.listen(source)
        try:
            text = r.recognize_google(audio_text, language=req_language)
            print('Converted audio transcripts into text: ', text)
            return text
        except:
            return "Could not understand your audio, PLease try again !"

@st.cache_resource()
def get_translator_obj():
    translator = Translator()
    return translator


def translate_to_english(text, speak_lang):
    req_language = "en"
    # match speak_lang:
    if speak_lang == "english":
        src_language = 'en'
    elif speak_lang == "hindi":
        src_language = 'hi'
    elif speak_lang == "telugu":
        src_language = 'te'
    elif speak_lang == "tamil":
        src_language = 'ta'
    elif speak_lang == "arabic":
        src_language = 'ar'
    else:
        src_language = 'en'

    translator = get_translator_obj()
    input_translations = translator.translate(text,
                                              dest='en', src=src_language)
    input_question_translated_to_eng = input_translations.text
    return input_question_translated_to_eng


def translate_eng_to_selected_lang(text, speak_lang):
    req_language = "en"
    # match speak_lang:
    if speak_lang == "english":
        src_language = 'en'
    elif speak_lang == "hindi":
        src_language = 'hi'
    elif speak_lang == "telugu":
        src_language = 'te'
    elif speak_lang == "tamil":
        src_language = 'ta'
    elif speak_lang == "arabic":
        src_language = 'ar'
    else:
        src_language = 'en'

    translator = get_translator_obj()
    output_translations = translator.translate(text,
                                               dest=src_language,
                                               src='en')
    return output_translations.text
