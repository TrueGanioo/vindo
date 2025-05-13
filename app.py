import streamlit as st
from analysis_and_model import analysis_and_model_page
from presentation import presentation_page

# Настройка конфигурации приложения (должен быть первым Streamlit-командой)
st.set_page_config(page_title="Predictive Maintenance App", layout="wide")

# Навигация с помощью streamlit-option-menu
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Analysis and Model", "Presentation"],
        icons=["bar-chart", "book"],
        menu_icon="cast",
        default_index=0
    )

# Вызов соответствующей страницы
if selected == "Analysis and Model":
    analysis_and_model_page()
elif selected == "Presentation":
    presentation_page()
