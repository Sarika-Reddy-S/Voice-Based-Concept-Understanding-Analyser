"""
main.py
-------
Entry-point alias for the VBCUA Streamlit application.

Allows running the app with:
    streamlit run main.py

as shown in the project documentation screenshots.
"""

# Simply re-export app.main so `streamlit run main.py` works identically.
from app import main

if __name__ == "__main__":
    main()
