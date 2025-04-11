import streamlit as st
import joblib
from io import StringIO
import lyrics
import re
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning) # Ignore version mismatch warnings from sklearn

### Title/Header ###
st.title("Sentiment Analyzer by Daniel Hu")
st.header("Results", divider="red")

### Sidebar ###
home = st.sidebar.page_link("app.py", label="Home", icon = "🏠")
results = st.sidebar.page_link("pages/results.py", label="Results", icon = "🚀")
code = st.sidebar.page_link("pages/codebase.py", label = "Codebase", icon = "🤖")

### Tabs in results ###
demo, songs, graphs= st.tabs(["Demo", "Music", "Graphs"])

model1 = joblib.load("models/model1.joblib") # Load model

### Demo page ###
demo.markdown("Write something!", )

# File upload option
uploaded_file = demo.file_uploader("Upload a text file")
if uploaded_file is not None: stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
else: stringio = StringIO()

# Text input option
text = demo.text_area("Paste or Edit Text", value=stringio.getvalue())

# Analysis
text_prediction = model1.predict({text})
if len(text)  > 0:
    if text_prediction[0] == 'negative': demo.write("Your text was negative")
    else: demo.write("Your text was positive")
    
    # Get feedback
    demo.write("Was our analysis accurate?")
    sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
    selected = demo.feedback("thumbs")
    
    # TODO: Add user input to dataset with its label
       
   
### Songs page ###
songs.markdown("Try it on a song!", )

songLeft, songRight = songs.columns(2) # Input/Analysis on left, lyrics on right

# Get song parameters  
artist = songLeft.text_input("Artist Name")
song = songLeft.text_input("Song Title")

if len(artist) > 0 and len(song) > 0:
    song_lyrics = lyrics.get_azlyrics(artist, song) # Get song


    formatted_lyrics = re.sub(r'([A-Z])', r'\n\1', song_lyrics) # Reformat lyrics text. Insert \n before every capital letter
    
    # Calculate number of lines
    lines = formatted_lyrics.split('\n')
    num_lines = len(lines)

    # Set text box height based on the number of lines
    height = num_lines * 20
    songRight.text_area("Lyrics (lesser known/newer songs may not appear)", value=formatted_lyrics, height = height)

    # Analysis
    song_prediction = model1.predict({formatted_lyrics})
    if len(formatted_lyrics)  > 0:
        if song_prediction[0] == 'negative': songLeft.write(f"'{song}' by {artist} is negative")
        else: songLeft.write(f"'{song}' by {artist} is positive")
     
### Graphs page ###
graphs.write("Visualization of the model's accuracy")
graph1, graph2 = graphs.columns(2)
graph1.image("images/sentimentBar.png")
graph2.image("images/sentimentCF.png")

