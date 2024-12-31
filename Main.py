# Analisis de sentimientos de reseñas de Amazon Gift Cards
# Nombre: Fernando Jesus Alejandre Ojeda
# Modulo: Línea de comandos en Python para Data Engineer
# Fecha: 30/12/2024

# Import necessary libraries
import json  # For working with JSON data
import re  # For regular expressions
import string  # For string manipulation
import nltk  # Natural Language Toolkit for text processing
from nltk.corpus import stopwords  # To remove common stop words
from nltk.tokenize import word_tokenize  # For tokenizing text
from Bert import analyze  # Custom BERT-based sentiment analysis module
import matplotlib.pyplot as plt  # For creating visualizations

# Import the Review class from Review module
from Review import Review as UserReview

def read_file(file_path):
    '''
    Reads a given file and converts it into a list of `Review` objects.

    Args:
        file_path (str): Path to the JSON file containing reviews.

    Returns:
        list: List of `Review` objects initialized with data from the file.
    '''
    reviews = []
    with open(file_path, 'r') as file:
        for line in file:
            review = json.loads(line.strip())  # Parse JSON from each line
            ur = UserReview()  # Create a new `Review` object
            ur.init_from_json(review)  # Initialize it with data from JSON
            reviews.append(ur)  # Add the object to the reviews list
    return reviews

def normalize_text(txt):
    '''
    Normalizes text by converting to lowercase, removing punctuation and numbers, and trimming whitespace.

    Args:
        txt (str): Input text to normalize.

    Returns:
        str: Normalized text.
    '''
    txt = txt.lower()  # Convert text to lowercase
    txt = txt.translate(str.maketrans('', '', string.punctuation + '¡¿'))  # Remove punctuation
    txt = re.sub(r'\d+', '', txt)  # Remove digits
    txt = re.sub(r'\s+', ' ', txt).strip()  # Remove extra whitespace
    return txt

def rem_token_and_stop_words(txt):
    '''
    Removes stop words from the text and tokenizes it.

    Args:
        txt (str): Input text.

    Returns:
        str: Text without stop words, joined back into a single string.
    '''
    tokens = word_tokenize(txt)  # Tokenize the text into words
    stop_words = set(stopwords.words('english'))  # Get the set of English stop words
    tokens_filtered = [word for word in tokens if word.lower() not in stop_words]  # Filter out stop words
    return ' '.join(tokens_filtered)  # Join the tokens back into a string

def process_reviews(reviews):
    '''
    Processes a list of reviews by normalizing text, removing stop words, and performing sentiment analysis.

    Args:
        reviews (list): List of `Review` objects to process.
    '''
    for review in reviews:
        text = review.comment  # Extract the review comment
        # Step 1 - Normalize the text
        text = normalize_text(text)
        # Step 2 - Remove stop words and tokenize
        tokens = rem_token_and_stop_words(text)
        # Step 3 - Perform sentiment analysis using BERT
        resultado = analyze(tokens)
        review.bert_grade = resultado  # Assign the BERT sentiment grade to the review
    return

if __name__ == '__main__':
    '''
    Main entry point of the program.

    This script processes Amazon Gift Card reviews and visualizes the sentiment analysis results.
    '''
    # Load reviews from the specified JSON file
    reviews = read_file("Resources/Gift_Cards_reviews.json")

    # Process the reviews: normalize, filter, and analyze sentiment
    process_reviews(reviews)
    
    # Prepare data for sentiment analysis visualization
    tags = ['insatisfecho', 'neutral', 'satisfecho']  # Sentiment categories
    bert_values = [0, 0, 0]  # Counters for BERT sentiment analysis
    rating_values = [0, 0, 0]  # Counters for user ratings

    for review in reviews:
        # Categorize based on BERT sentiment grade
        if review.bert_grade < 3:
            bert_values[0] += 1  # Insatisfecho
        elif review.bert_grade == 3:
            bert_values[1] += 1  # Neutral
        else:
            bert_values[2] += 1  # Satisfecho

        # Categorize based on user rating
        if review.rating < 3:
            rating_values[0] += 1  # Insatisfecho
        elif review.rating == 3:
            rating_values[1] += 1  # Neutral
        else:
            rating_values[2] += 1  # Satisfecho

    # Visualization: Sentiment analysis by BERT
    plt.bar(tags, bert_values)
    plt.title('Análisis de reseñas Amazon por BERT')
    plt.xlabel('Etiqueta')
    plt.ylabel('Cantidad')
    plt.show()
    
    # Visualization: Sentiment analysis by user rating
    plt.bar(tags, rating_values)
    plt.title('Análisis de reseñas Amazon por Rating')
    plt.xlabel('Etiqueta')
    plt.ylabel('Cantidad')
    plt.show()