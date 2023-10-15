# Import necessary packages
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Read the URL file into a pandas DataFrame
df = pd.read_excel('Input.xlsx')

# Initialize lists to store text analysis results
positive_score = []
negative_score = []
polarity_score = []
subjectivity_score = []
avg_sentence_length = []
percentage_of_complex_words = []
fog_index = []
complex_word_count = []
avg_syllable_word_count = []
word_count = []
average_word_length = []
pp_count = []

# Function to measure various text features
def measure(file):
    with open(os.path.join(text_dir, file), 'r') as f:
        text = f.read()
        text = re.sub(r'[^\w\s]', '', text)
        sentences = text.split('.')
        num_sentences = len(sentences)
        words = [word for word in text.split() if word.lower() not in stopwords]
        num_words = len(words)
        
        complex_words = [word for word in words if len(re.findall(r'[aeiouyAEIOUY]{3,}', word)) > 2]
        num_complex_words = len(complex_words)
        
        avg_sentence_len = num_words / num_sentences
        percentage_complex_words = num_complex_words / num_words
        fog = 0.4 * (avg_sentence_len + percentage_complex_words)
        
        syllable_count = 0
        syllable_words = []
        for word in words:
            if word.endswith('es'):
                word = word[:-2]
            elif word.endswith('ed'):
                word = word[:-2]
            syllables = len(re.findall(r'[aeiouyAEIOUY]{1,2}', word))
            syllable_count += syllables
            if syllables > 2:
                syllable_words.append(word)
        
        avg_syllable_word = syllable_count / len(syllable_words)
        
        pp_count = len(re.findall(r'\b(?:I|we|my|ours|us)\b', text))
        
        length = sum(len(word) for word in words)
        avg_word_len = length / len(words)
        
        return avg_sentence_len, percentage_complex_words, fog, num_complex_words, avg_syllable_word, len(words), avg_word_len, pp_count

# Set directory paths
text_dir = "/gdrive/MyDrive/project/Data_Extraction_and_NLP/TestAssignment/TitleText"
stopwords_dir = "/gdrive/MyDrive/project/Data_Extraction_and_NLP/TestAssignment/StopWords"
sentiment_dir = "/gdrive/MyDrive/project/Data_Extraction_and_NLP/TestAssignment/MasterDictionary"

# Load stop words from the stopwords directory and store in a set
stop_words = set()
for filename in os.listdir(stopwords_dir):
    with open(os.path.join(stopwords_dir, filename), 'r', encoding='ISO-8859-1') as f:
        stop_words.update(set(f.read().splitlines()))

# Load text files from the directory and tokenize them
docs = []
for text_file in os.listdir(text_dir):
    with open(os.path.join(text_dir, text_file), 'r') as f:
        text = f.read()
        words = word_tokenize(text)
        filtered_text = [word for word in words if word.lower() not in stop_words]
        docs.append(filtered_text)

# Store positive and negative words from the sentiment directory
positive_words = set()
negative_words = set()

for filename in os.listdir(sentiment_dir):
    with open(os.path.join(sentiment_dir, filename), 'r', encoding='ISO-8859-1') as f:
        if filename == 'positive-words.txt':
            positive_words.update(f.read().splitlines())
        else:
            negative_words.update(f.read().splitlines())

# Calculate sentiment scores for each document
for doc in docs:
    positive_score.append(len([word for word in doc if word.lower() in positive_words]))
    negative_score.append(len([word for word in doc if word.lower() in negative_words]))
    total_words = len(doc)
    polarity_score.append((positive_score[-1] - negative_score[-1]) / (total_words + 1e-6))
    subjectivity_score.append((positive_score[-1] + negative_score[-1]) / (total_words + 1e-6))

# Measure text features for each document
for file in os.listdir(text_dir):
    (avg_sent_len, perc_complex_words, fog, num_complex, avg_syllable, num_words, avg_word_len, pp) = measure(file)
    avg_sentence_length.append(avg_sent_len)
    percentage_of_complex_words.append(perc_complex_words)
    fog_index.append(fog)
    complex_word_count.append(num_complex)
    avg_syllable_word_count.append(avg_syllable)
    word_count.append(num_words)
    average_word_length.append(avg_word_len)
    pp_count.append(pp)

# Create a new DataFrame for the output data
output_df = pd.read_excel('Output Data Structure.xlsx')

# Drop rows where URL_ID 44, 57, and 144 do not exist (404 errors)
output_df.drop([44 - 37, 57 - 37, 144 - 37], axis=0, inplace=True)

# Assign the calculated values to the appropriate columns in the output DataFrame
output_df['Positive_Score'] = positive_score
output_df['Negative_Score'] = negative_score
output_df['Polarity_Score'] = polarity_score
output_df['Subjectivity_Score'] = subjectivity_score
output_df['Avg_Sentence_Length'] = avg_sentence_length
output_df['Percentage_of_Complex_Words'] = percentage_of_complex_words
output_df['Fog_Index'] = fog_index
output_df['Complex_Word_Count'] = complex_word_count
output_df['Word_Count'] = word_count
output_df['Avg_Syllable_Per_Word'] = avg_syllable_word_count
output_df['Average_Word_Length'] = average_word_length
output_df['Personal_Pronoun_Count'] = pp_count

# Save the output DataFrame to a CSV file
output_df.to_csv('OutputAssignment.csv', index=False)
