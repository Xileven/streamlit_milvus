import streamlit as st

st.set_page_config(
    page_title="Chat with your Documents",
    page_icon=":books:"
)

st.title("Chat with your Documents")
st.subheader("BAMA, Feb 2025")

st.markdown("""
**Sample Questions:**
```
- [PDF] How Integration of Ameritrade impact client metrics from 2023 to 2024?
- [Excel] Where is the headquarters of schwab and what is its size, including leased and owned
- [PDF & Word] Compare Client Metrics of Three Month Ended from 2022, to 2023, to 2024, in numbers, and printout in table
- [PDF & Word] Compare Total Net Revenue from 2022, to 2023, to 2024 and printout in table
- [Summary] based on Client Metrics of Three Month Ended from 2022, to 2023, to 2024, analyze the business
- Compare Total Net Revenue from 2022, to 2023, to 2024 and printout in table
```
""")

import sys
from pathlib import Path

if __name__ == "__main__":
    # Remove the CWD from sys.path while we load stuff.
    # This is added back by InteractiveShellApp.init_path()
    if sys.path[0] == "" or Path(sys.path[0]) == Path.cwd():
        del sys.path[0]

    from ipykernel import kernelapp as app

    app.launch_new_instance()
