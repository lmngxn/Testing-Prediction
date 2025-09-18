import streamlit as st
import random
import os

# """
# This file contains the common components used in the Streamlit App.
# This includes the sidebar, the title, the footer, and the password check.
# """

def check_password():
    """Returns `True` if the user has the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if os.getenv("PASSWORD") == st.session_state.get("password"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Check if the PASSWORD environment variable is set
    password_env = os.getenv("PASSWORD")
    if password_env is None or password_env == "":
        return True  # Skip password check if not set

    # If the password has already been validated, return True
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )

    # Show error if the password is incorrect
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect")

    return False
