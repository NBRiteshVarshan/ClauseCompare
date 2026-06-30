import os
import sys
import streamlit.web.cli as stcli
import webbrowser
from threading import Timer

def open_browser():
    webbrowser.open("http://localhost:8501")

if __name__ == "__main__":
    if getattr(sys, 'frozen', False):
        current_dir = sys._MEIPASS
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Targeting your specific app.py file
    target_script = os.path.join(current_dir, "app.py")
    
    Timer(2.0, open_browser).start()
    
    sys.argv = ["streamlit", "run", target_script, "--global.developmentMode=false"]
    sys.exit(stcli.main())
