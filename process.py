from cmath import log
import spacy
import re
import string
import textwrap
from fpdf import FPDF
from logger import Logger
import os
import base64
import streamlit as st
#from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords # Import stopwords from nltk.corpus
from nltk.stem import WordNetLemmatizer
import en_core_web_sm
from nltk.corpus import wordnet as wn
import pandas as pd
from rake_nltk import Rake
import pytextrank
from sklearn.decomposition import TruncatedSVD
import gensim
from gensim import corpora,models
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize

file = open("log.txt", "a+")
logger = Logger()

def preprocessing(text):
    logger.log(file, f"starting preprocessing")
    # Make lower
    text = text.lower()

    # Remove line breaks
    text = re.sub(r'\n', ' ', text)
    # Remove line breaks
    text = re.sub(r'\t', '', text)

    text = re.sub("[^A-Za-z0-9\s\.\,]+"," ", text)

    text = re.sub(r'[0-9]', ' ', text)

    text = text.split()

    with open(os.path.join("stopwords.txt"),'r') as useless_words:
        lis = useless_words.read().split("\n")
        try:
            stop_words = stopwords.words('english')
            logger.log(file, f"trying to load eng stopwords from model")

        except:
            logger.log(file, f"load failed downloading stopwords from nlkt")
            nltk.download('stopwords')
            stop_words = stopwords.words('english')
            lis = set(lis + stop_words)
        finally:
            lis = list(lis) + ['hi', 'im']

            try:
                logger.log(file, f"trying loading wordlemma")
                lem = WordNetLemmatizer()
                lem.lemmatize("testing")
            except:
                logger.log(file, f"loading failed trying to download wordnetm and omw 1.4")
                #call the nltk downloader
                nltk.download('wordnet')
                nltk.download('omw-1.4')
                lem = WordNetLemmatizer() #stemming
            finally:
                logger.log(file, f"lemmatize words preprocessing done")
                text_filtered = [lem.lemmatize(word) for word in text if not word in lis]
                return " ".join(text_filtered)

def text_process_lsa(text):
    text = preprocessing(text)
    data = extract_keywords_lsa(text)
    logger.log(file, f"text rank done")
    data = ", \n".join(str(d) for d in data)

    if data == "":
        data = "None Keyword Found"
    logger.log(file, "data cleaned and returned")
    return data

def text_process_lda(text):
    text = preprocessing(text)
    data = extract_keywords_lda(text)
    logger.log(file, f"text rank done")
    data = ", \n".join(str(d) for d in data)

    if data == "":
        data = "None Keyword Found"
    logger.log(file, "data cleaned and returned")
    return data

def text_to_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    
    line_height = pdf.font_size * 1.5

    # Read file content if 'text' is a file object
    if hasattr(text, 'read'):
        text = text.read()

    # Split the text into lines
    lines = text.split('\n')

    for line in lines:
        # Split long lines into multiple lines if they exceed the page width
        pdf.multi_cell(0, line_height, line)
    
    pdf.output(filename)
    
    logger.log(file, "PDF File saved")

def text_doc(file, filename):
    # Read the content from the input file
    content = file.read()
    
    # Define the structure of the Word document
    word_document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
        <w:body>
            <w:p>
                <w:r>
                    <w:t>{content}</w:t>
                </w:r>
            </w:p>
        </w:body>
    </w:document>"""

    # Create the required folders and files for a .docx file
    os.makedirs(f'{filename}_files/word', exist_ok=True)
    os.makedirs(f'{filename}_files/_rels', exist_ok=True)
    os.makedirs(f'{filename}_files/docProps', exist_ok=True)
    
    # Write the content to the document.xml file
    with open(f'{filename}_files/word/document.xml', 'w', encoding='utf-8') as f:
        f.write(word_document)
    
    # Create the other necessary files
    with open(f'{filename}_files/[Content_Types].xml', 'w', encoding='utf-8') as f:
        f.write('''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
                <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
                <Default Extension="xml" ContentType="application/xml"/>
                <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
            </Types>''')

    with open(f'{filename}_files/_rels/.rels', 'w', encoding='utf-8') as f:
        f.write('''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
            </Relationships>''')

    with open(f'{filename}_files/docProps/core.xml', 'w', encoding='utf-8') as f:
        f.write('''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                <dc:title></dc:title>
                <dc:subject></dc:subject>
                <dc:creator></dc:creator>
                <cp:keywords></cp:keywords>
                <dc:description></dc:description>
                <cp:lastModifiedBy></cp:lastModifiedBy>
                <cp:revision>1</cp:revision>
                <dcterms:created xsi:type="dcterms:W3CDTF"></dcterms:created>
                <dcterms:modified xsi:type="dcterms:W3CDTF"></dcterms:modified>
            </cp:coreProperties>''')

    # Create the .docx file
    with zipfile.ZipFile(f'{filename}.docx', 'w') as docx:
        docx.write(f'{filename}_files/word/document.xml', 'word/document.xml')
        docx.write(f'{filename}_files/[Content_Types].xml', '[Content_Types].xml')
        docx.write(f'{filename}_files/_rels/.rels', '_rels/.rels')
        docx.write(f'{filename}_files/docProps/core.xml', 'docProps/core.xml')
    
    # Clean up the temporary files
    for root, dirs, files in os.walk(f'{filename}_files', topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(f'{filename}_files')



def extract_keywords_lsa(text, num_keywords=10):
    # Preprocess the text
    processed_text = preprocessing(text)

    # Convert text to TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_text])

    # Apply SVD
    svd = TruncatedSVD(n_components=1)
    svd_matrix = svd.fit_transform(tfidf_matrix)

    # Get terms and their corresponding scores
    terms = vectorizer.get_feature_names_out()
    scores = svd.components_[0]

    # Create a dataframe of terms and scores
    term_scores = pd.DataFrame({'term': terms, 'score': scores})

    # Sort terms by score and select top N terms
    top_keywords = term_scores.sort_values(by='score', ascending=False).head(num_keywords)

    return top_keywords['term'].tolist()

def extract_keywords_lda(text, num_keywords=10):
    # Preprocess the text
    processed_text = preprocessing(text)

    # Tokenize the text
    tokens = processed_text.split()

    # Create dictionary and corpus
    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]

    # Apply LDA model
    lda_model = models.LdaModel(corpus, num_topics=1, id2word=dictionary)

    # Get the topic distribution for the document
    topic_distribution = lda_model[corpus][0]

    # Sort topics by score and select top topic
    top_topic = sorted(topic_distribution, key=lambda x: x[1], reverse=True)[0][0]

    # Get top keywords for the selected topic
    top_keywords = lda_model.print_topic(top_topic, topn=num_keywords).split('+')

    # Extracting keywords from the formatted output
    keywords = [word.split('*')[1].replace('"', '').strip() for word in top_keywords]

    return keywords
