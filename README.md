# Keyword Extraction and Topic Modelling App

This Streamlit application provides functionalities to analyze and summarize text. It offers a user-friendly interface to perform text analysis and summarization tasks, logging all actions for future reference.

Keywords summarize a document concisely and give a high-level description of the document’s content and Keyword extraction is a process in which you extract important keywords and phrases for analysis of texts and documents. Topic modelling is the process of categorizing text into a given topic which gives us an idea of what it is about without reading the whole text. This is very important as it saves time and give people a context.

## Features

- **Keyword Extraction**: Extracts relevant keywords from a given text using Natural Language Processing (NLP)
- **Topic Modelling**: Identifies underlying topics in a collection of texts using Latent Dirichlret Allocation and Latent Semantic Analysis
- **Text Summarization**: Generates a concise summary of a given text using TextRank and LexRank algorithm
- **Email Spam Classifier**: Classify if a given email is spam or not.
- **Sentiment Analysis**: Analyze the sentiment of a given text as positive or negative 
- **WordCloud Generator**: Prepares a text representation in which words are shown in varying sizes depending on how often they appear in our corpus
- **N Gram Analysis**: Identify the most commonly occurring n-grams. While word cloud focusses on singular words, it yields multiple phrases instead of just one.
- **Named Entity Recognition**: Classifies text into pre-defined categories such as the names of persons, organizations, locations, expressions of times, quantities, monetary values etc.

## Installation

To run this application locally, follow these steps:

1. Clone the repository
2. To install all the libraries (preferably in a VM): pip install -r requirements.txt
3. To run the app: streamlit run app.py

Or you could run it locally on your system using : https://keywordextractiontopicmodelling.streamlit.app/
