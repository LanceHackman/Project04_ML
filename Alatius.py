import requests
from bs4 import BeautifulSoup

"""
Used HTML algorithm on an online macronizer
"""
def get_scansion_from_alatius(latin_line):
    """
    Get scansion by scraping the macronizer page
    Verifies the result is for the correct input line
    """
    # The URL for the web script, based on the default setup
    URL = "https://alatius.com/macronizer/"

    # The text and options you want to send
    text_to_macronize = latin_line

    # The data dictionary mimics the web form submission:
    data = {
        'textcontent': text_to_macronize,
        'macronize': 'on',  # Mark long vowels
        'alsomaius': '',  # Do not mark maius etc.
        'scan': 1,  # 1 for dactylic hexameters
        'doevaluate': '',  # Do not evaluate
        'utov': '',  # Do not convert u to v
        'itoj': ''  # Do not convert i to j
    }

    try:
        # Send a POST request with the form data
        response = requests.post(URL, data=data)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Parse the resulting HTML page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the Scansion Feet (DDSSDS)
        # This targets the div with class 'feet' from the HTML source
        feet_div = soup.find('div', class_='feet')
        scanned_feet = feet_div.get_text().replace('\n', ' ') if feet_div else "Scansion feet not found."

        # Extract the Macronized Text
        # This targets the div with id 'selectme'
        text_div = soup.find('div', id='selectme')
        # Use .text to get the clean text without the HTML span tags
        macronized_text = text_div.text.strip() if text_div else "Macronized text not found."

        # Print the results
        print("--- Scraper Results ---")
        print(f"Text Sent: {text_to_macronize}")
        print(f"Scansion Feet: {scanned_feet}")
        print(f"Macronized Text: {macronized_text}")

        return scanned_feet

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None