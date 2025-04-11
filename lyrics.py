import requests
from bs4 import BeautifulSoup

def get_azlyrics(artist, song):
    # Format for AZLyrics URL
    artist = artist.lower().replace(" ", "")
    song = song.lower().replace(" ", "")
    url = f"https://www.azlyrics.com/lyrics/{artist}/{song}.html"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Error fetching lyrics: {e}"

    soup = BeautifulSoup(response.text, "html.parser")

    # Find all divs that don't have a class and aren't within script/style tags
    divs = soup.find_all("div")
    for div in divs:
        if not div.attrs and "Usage of azlyrics.com" not in div.text:
            return div.get_text(strip=True)
    
    return "Lyrics not found."
