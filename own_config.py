
import streamlit as st

UPLOAD_URLS_LIST = []
if UPLOAD_URLS_LIST not in st.session_state:
    st.session_state[UPLOAD_URLS_LIST] = []

APP_IN_EXECUTION_STATE = False
if APP_IN_EXECUTION_STATE not in st.session_state:
    st.session_state[APP_IN_EXECUTION_STATE] = False