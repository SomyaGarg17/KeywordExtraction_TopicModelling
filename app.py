import streamlit as st
import warnings
from PIL import  Image
import numpy as np
from io import StringIO 
import docx2txt
from logger import Logger
import os
from streamlit_quill import st_quill
from process import text_process_lsa, text_to_pdf, text_doc, text_process_lda
import librosa
import gensim
from gensim import corpora, models
import re
import heapq
import whisper
from pytube import YouTube
from pathlib import Path
import pandas as pd
import spam_filter as sf
import text_analysis as nlp
import text_summarize as ts
import joblib
from rake_nltk import Rake
import pprint
import tempfile
from io import StringIO
from PIL import  Image
import spacy
import spacy_streamlit
from collections import Counter
import en_core_web_sm
from nltk.tokenize import sent_tokenize
import nltk


# Warnings ignore 
warnings.filterwarnings(action='ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config("Keyword Extraction and Topic Modelling",layout="wide",menu_items={'Get Help': 'https://www.extremelycoolapp.com/help','Report a bug': "https://www.extremelycoolapp.com/bug",'About': "# This is an *extremely* cool app for keyword extraction and topic modelling!"})

#### topic labeling ##########
insurance_keywords = ['actuary', 'claims', 'coverage', 'deductible', 'policyholder', 'premium', 'underwriter', 
                      'risk assessment', 'insurable interest', 'loss ratio', 'reinsurance', 'actuarial tables', 
                      'property damage', 'liability', 'flood insurance', 'term life insurance', 
                      'whole life insurance', 'health insurance', 'auto insurance', 'homeowners insurance', 
                      'marine insurance', 'crop insurance', 'catastrophe insurance', 'umbrella insurance', 
                      'pet insurance', 'travel insurance', 'professional liability insurance', 
                      'disability insurance', 'long-term care insurance', 'annuity', 'pension plan', 
                      'group insurance', 'insurtech', 'insured', 'insurer', 'subrogation', 'adjuster', 
                      'third-party administrator', 'excess and surplus lines', 'captives', 'workers compensation', 
                      'insurance fraud', 'health savings account', 'health maintenance organization', 
                      'preferred provider organization']

finance_keywords = ['assets', 'liability', 'equity', 'capital', 'portfolio', 'dividend', 'financial statement', 
                    'balance sheet', 'income statement', 'cash flow statement', 'statement of retained earnings', 
                    'financial ratio', 'valuation', 'bond', 'stock', 'mutual fund', 'exchange-traded fund', 
                    'hedge fund', 'private equity', 'venture capital', 'mergers and acquisitions', 
                    'initial public offering', 'secondary market', 'primary market', 'securities', 'derivative', 
                    'option', 'futures', 'forward contract', 'swaps', 'commodities', 'credit rating', 
                    'credit score', 'credit report', 'credit bureau', 'credit history', 'credit limit', 
                    'credit utilization', 'credit counseling', 'credit card', 'debit card', 'ATM', 'bankruptcy', 
                    'foreclosure', 'debt consolidation', 'taxes', 'tax return', 'tax deduction', 'tax credit', 
                    'tax bracket', 'taxable income']

banking_capital_markets_keywords = ['bank', 'credit union', 'savings and loan association', 'commercial bank', 
                                    'investment bank', 'retail bank', 'wholesale bank', 'online bank', 
                                    'mobile banking', 'checking account', 'savings account', 'money market account',
                                    'certificate of deposit', 'loan', 'mortgage', 'home equity loan', 
                                    'line of credit', 'credit card', 'debit card', 'ATM', 'automated clearing house', 
                                    'wire transfer', 'ACH', 'SWIFT', 'international banking', 'foreign exchange', 
                                    'forex', 'currency exchange', 'central bank', 'Federal Reserve', 'interest rate', 
                                    'inflation', 'deflation', 'monetary policy', 'fiscal policy', 
                                    'quantitative easing', 'securities', 'stock', 'bond', 'mutual fund', 
                                    'exchange-traded fund', 'hedge fund', 'private equity', 'venture capital', 
                                    'investment management', 'portfolio management', 'wealth management', 
                                    'financial planning']

healthcare_life_sciences_keywords = ['medical device', 'pharmaceutical', 'biotechnology', 'clinical trial', 'FDA', 
                                     'healthcare provider', 'healthcare plan', 'healthcare insurance', 'patient', 
                                     'doctor', 'nurse', 'pharmacist', 'hospital', 'clinic', 'healthcare system', 
                                     'healthcare policy', 'public health', 'healthcare IT', 
                                     'electronic health record', 'telemedicine', 'personalized medicine', 
                                     'genomics', 'proteomics', 'clinical research', 'drug development', 
                                     'drug discovery', 'medicine', 'health']

law_keywords = ['law', 'legal', 'attorney', 'lawyer', 'litigation', 'arbitration', 'dispute resolution', 
                'contract law', 'intellectual property', 'corporate law', 'labor law', 'tax law', 
                'real estate law', 'environmental law', 'criminal law', 'family law', 'immigration law', 
                'bankruptcy law']

sports_keywords = ['sports', 'football', 'basketball', 'baseball', 'hockey', 'soccer', 'golf', 'tennis', 
                   'olympics', 'athletics', 'coaching', 'sports management', 'sports medicine', 'sports psychology', 
                   'sports broadcasting', 'sports journalism', 'esports', 'fitness']

media_keywords = ['media', 'entertainment', 'film', 'television', 'radio', 'music', 'news', 'journalism', 
                  'publishing', 'public relations', 'advertising', 'marketing', 'social media', 'digital media', 
                  'animation', 'graphic design', 'web design', 'video production']

manufacturing_keywords = ['manufacturing', 'production', 'assembly', 'logistics', 'supply chain', 
                          'quality control', 'lean manufacturing', 'six sigma', 'industrial engineering', 
                          'process improvement', 'machinery', 'automation', 'aerospace', 'automotive', 
                          'chemicals', 'construction materials', 'consumer goods', 'electronics', 'semiconductors']

automotive_keywords = ['automotive', 'cars', 'trucks', 'SUVs', 'electric vehicles', 'hybrid vehicles', 
                       'autonomous vehicles', 'car manufacturing', 'automotive design', 'car dealerships', 
                       'auto parts', 'vehicle maintenance', 'car rental', 'fleet management', 'telematics']

telecom_keywords = ['telecom', 'telecommunications', 'wireless', 'networks', 'internet', 'broadband', 
                    'fiber optics', '5G', 'telecom infrastructure', 'telecom equipment', 'VoIP', 
                    'satellite communications', 'mobile devices', 'smartphones', 'telecom services', 
                    'telecom regulation', 'telecom policy']

information_technology_keywords = [
    "Artificial intelligence", "Machine learning", "Data Science", "Big Data", "Cloud Computing",
    "Cybersecurity", "Information security", "Network security", "Blockchain", "Cryptocurrency",
    "Internet of things", "IoT", "Web development", "Mobile development", "Frontend development",
    "Backend development", "Software engineering", "Software development", "Programming",
    "Database", "Data analytics", "Business intelligence", "DevOps", "Agile", "Scrum",
    "Product management", "Project management", "IT consulting", "IT service management", 
    "ERP", "CRM", "SaaS", "PaaS", "IaaS", "Virtualization", "Artificial reality", "AR", "Virtual reality",
    "VR", "Gaming", "E-commerce", "Digital marketing", "SEO", "SEM", "Content marketing",
    "Social media marketing", "User experience", "UX design", "UI design", "Cloud-native",
    "Microservices", "Serverless", "Containerization"
]

industries = {
    'Insurance': insurance_keywords,
    'Finance': finance_keywords,
    'Banking': banking_capital_markets_keywords,
    'Healthcare': healthcare_life_sciences_keywords,
    'Legal': law_keywords,
    'Sports': sports_keywords,
    'Media': media_keywords,
    'Manufacturing': manufacturing_keywords,
    'Automotive': automotive_keywords,
    'Telecom': telecom_keywords,
    'IT': information_technology_keywords
}

def save_to_file(str_data, readmode = "w"):
    
    if readmode == "w":
        with open(os.path.join("userdata.txt"), readmode) as file_obj:
            file_obj.write(str_data)
    
    else:
        st.session_state['user_data'] = 1
        with open(os.path.join("userdata.txt"), readmode) as file_obj:
            file_obj.write(str_data)

    logger.log(file, "file saved")

def process_data(uploaded_file) -> str:
       
        try:
            data = docx2txt.process(uploaded_file)
            logger.log(file, "data processed to str")
            return data
       
        except KeyError as e:
            logger.log(file, f"data processing failed: {e}")
            return None

def get_doc(uploaded_file):
   
    if uploaded_file is not None:
       
        if st.button("Proceed"):
            str_data = process_data(uploaded_file)
         
            if str_data:
                st.subheader('Edit Data')
                st.session_state['str_value'] = str_data
                logger.log(file, "updated data to session from doc string")
                st.session_state['load_editor'] = True
                return str_data
            
            else:
                st.subheader('File Corrupted please upload other file')
                return str_data

def run_editor(str_data, key = "editor"):
   
    # Spawn a new Quill editor
    logger.log(file, "starting editor")
    content = st_quill(value = str_data,key=key)
    st.session_state['str_value'] = content
    logger.log(file, "returning editor new content")
   
    return content

def label_topic(text):
    
    # Count the number of occurrences of each keyword in the text for each industry
    counts = {}
    for industry, keywords in industries.items():
        count = sum(1 for keyword in keywords if re.search(r"\b{}\b".format(keyword), text, re.IGNORECASE))
        counts[industry] = count

       # If no industry has a count, return None
    non_zero_counts = {industry: count for industry, count in counts.items() if count > 0}

      # If no industry has a non-zero count, return None
    if not non_zero_counts:
        return None
    

    top_industries = heapq.nlargest(2, non_zero_counts, key=non_zero_counts.get)

    # If only one industry was found, return it
    if len(top_industries) == 1:
        return top_industries[0]
    
    # If two industries were found, return them both
    return top_industries

#This function performs topic modelling including steps like preprocessing text and using LDA Model 
def perform_topic_modeling(transcript_text, num_topics=5, num_words=10):
    # Preprocess the transcript text
    preprocessed_text = preprocess_text(transcript_text)

    # Create a dictionary of all unique words in the transcripts
    dictionary = corpora.Dictionary(preprocessed_text)

    # Convert the preprocessed transcripts into a bag-of-words representation
    corpus = [dictionary.doc2bow(text) for text in preprocessed_text]

    # Train an LDA model with the specified number of topics
    lsi_model = models.LsiModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

    # Extract the most probable words for each topic
    topics = []
    for idx, topic in lsi_model.print_topics(-1, num_words=num_words):
        # Extract the top words for each topic and store in a list
        topic_words = [word.split('*')[1].replace('"', '').strip() for word in topic.split('+')]
        topics.append((f"Topic {idx}", topic_words))

    return topics

def perform_topic_modeling_LDA(transcript_text, num_topics=5, num_words=10):
   
    # Preprocess the transcript text
    preprocessed_text = preprocess_text(transcript_text)

    # Create a dictionary of all unique words in the transcripts
    dictionary = corpora.Dictionary(preprocessed_text)

    # Convert the preprocessed transcripts into a bag-of-words representation
    corpus = [dictionary.doc2bow(text) for text in preprocessed_text]

    # Train an LDA model with the specified number of topics
    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

    # Extract the most probable words for each topic
    topics = []
    for idx, topic in lda_model.print_topics(-1, num_words=num_words):
        # Extract the top words for each topic and store in a list
        topic_words = [word.split('*')[1].replace('"', '').strip() for word in topic.split('+')]
        topics.append((f"Topic {idx}", topic_words))

    return topics

#This function is used to preprocess the text
def preprocess_text(text):
   
    #It simply tokenizes the text and removes stop words
    tokens = gensim.utils.simple_preprocess(text)
    stop_words = gensim.parsing.preprocessing.STOPWORDS
    preprocessed_text = [[token for token in tokens if token not in stop_words]]

    return preprocessed_text

# This function is decorated with @st.cache_resource, 
 #   which means that the result of this function will be cached and reused 
  #  if the function is called again with the same arguments.

@st.cache_resource
#This function is responsible for loading a pre-trained Whisper model for automatic speech recognition 
def load_model():
   
    model = whisper.load_model("base")
  
    return model

#This function is responsible for downloading a YouTube video and saving it 
#as an MP4 file
def save_video(url):
   
    try:
        # Create YouTube object
        yt = YouTube(url)        
        # Get the first available stream
        stream = yt.streams.first()
        # Define the filename with the video title and .mp4 extension
        filename = f"{yt.title}.mp4"
        # Get the current working directory
        output_path = os.getcwd()
        # Combine the output path and filename to get the full file path
        file_path = os.path.join(output_path, filename)
        # Download the video to the specified path
        stream.download(filename=filename)
        # Return the video title, filename, and full file path
        return yt.title, filename, file_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

#This function converts the video into transcript
def video_to_transcript(video_file):
    
    model = load_model()
    video_file = Path(video_file)
    if not video_file.exists():
        raise FileNotFoundError(f"File '{video_file}' not found.")
    
    video,sr = librosa.load(video_file.as_posix(), sr=16000)
    result = model.transcribe(video)
    transcript = result["text"]
   
    return transcript
    

#Describing the Web Application

# Title of the application 
st.title('Keyword Extraction and Topic Modelling App\n', )
st.write("Using Latent Semantic Analysis and Latent Dirichlet Allocation")

display = Image.open('images/keywords-extraction.jpg')
display = np.array(display)
st.image(display,width=1000)

# Sidebar options
option = st.sidebar.selectbox('Navigation', 
                              ["Home","Keyword Extractor", "Topic Modelling and Labelling","Email Spam Classifier", 
                                "Keyword Sentiment Analysis", "Word Cloud", "N-Gram Analysis", "Named Entity Recognition",
                                "Text Summarizer"])

if option == 'Home':
	st.write(
			"""
				## Project Description
				An application that leverages advanced natural language processing techniques for efficient keyword extraction and topic modeling, enabling users to automatically identify key terms and underlying themes in large text datasets. Ideal for text analysis, research, and data-driven insights.
			"""
		)
    
elif option=="Keyword Extractor":

    file = open("log.txt", "a+")
    logger = Logger()
    
    if "load_state" not in st.session_state:
        st.session_state['load_state'] = False
        st.session_state['load_editor'] = False
        st.session_state['str_value'] = None

    if __name__ == '__main__':
        st.session_state['user_data'] = 0
        st.session_state['load_state'] = True
        boundary = "\n"*4 + "=====Keywords======" + "\n"*4

        st.title("Keyword Extractor")
        st.caption("An application that automatically extracts key terms from large text documents using advanced algorithms. Ideal for improving content analysis and search functionality.")
        st.write("\n")
        st.subheader("Upload File")

        logger.log(file, "init done")
        uploaded_file = st.file_uploader("Upload Doc or Docx File Only",type = [".doc","docx"])
        str_data = get_doc(uploaded_file)

    if str_data or st.session_state['load_editor']:
        data = run_editor(str_data)

    if st.session_state['str_value'] is not None:
       
        if st.button("Save & Extract") or st.session_state['load_state']:
            logger.log(file, "Saving userdata")
            data = data + boundary
            save_to_file(data)
            model_select = st.selectbox("Model Selection", ["Latent Semantic Analysis", "Latent Dirichlet Allocation"])        
           
            if model_select == "Latent Semantic Analysis":
                logger.log(file, "user edited data saved. no extracting data")
                save_to_file(text_process_lsa(data), readmode="a+")
                logger.log(file, "data extracted and appended to the original userdata")
           
            else:
                logger.log(file, "user edited data saved. no extracting data")
                save_to_file(text_process_lda(data), readmode="a+")
                logger.log(file, "data extracted and appended to the original userdata")

            if st.session_state['user_data']:    
           
                if st.checkbox("Accept Terms & Condition"):
                    genre = st.radio("Download as",('PDF', 'DOC'))
                   
                    with open(os.path.join("userdata.txt"), 'r', encoding="latin-1") as df:
                        
                        if genre == 'PDF':
                            text_to_pdf(df, 'keywords.pdf')
                           
                            with open(os.path.join("keywords.pdf"), "rb") as pdf_file:
                                PDFbyte = pdf_file.read()
                                st.download_button(label="Export as PDF",data=PDFbyte,file_name="keywords.pdf",mime='application/octet-stream')

                        else:
                            text_doc(df, 'keywords')
                          
                            with open(os.path.join("keywords.doc"), "rb") as doc_file:
                                docbyte = doc_file.read()
                                st.download_button(label="Export as DOC",data=docbyte,file_name="keywords.doc",mime='application/octet-stream')

elif option=="Topic Modelling and Labelling":
   
    st.header("Topic Modelling and Labelling App")
    choice = st.selectbox("Select your choice", ["On Text", "On Video", "On CSV"])
   
    # Perform analysis based on the selected choice
    if choice == "On Text":
        st.subheader("Topic Modeling and Labeling on Text")
        
         # Create a text area widget to allow users to paste transcripts
        text_input = st.text_area("Paste enter text below", height=400)
       
        # Model Selection 
        model_select = st.selectbox("Model Selection", ["Latent Semantic Analysis", "Latent Dirichlet Allocation"])        
        if text_input is not None:
           
            if st.button("Analyze Text"):
           
                if model_select == "Latent Semantic Analysis":
                    col1, col2, col3 = st.columns([1,1,1])
           
                    with col1:
                        st.info("Text is below")
                        #Prints the text as the user provided
                        st.success(text_input)

                    with col2:
                        # Perform topic modeling on the transcript text
                        topics = perform_topic_modeling(text_input)
                        # Display the resulting topics in the app
                        st.info("Topics in the Text")
                        for topic in topics:
                            st.success(f"{topic[0]}: {', '.join(topic[1])}")

                    with col3:
                        st.info("Topic Labeling")
                        labeling_text = text_input
                        #Performs topic Labeling
                        industry = label_topic(labeling_text)
                        st.markdown("**Topic Labeling Industry Wise**")
                        st.write(industry)
          
                else:
                    col1, col2, col3 = st.columns([1,1,1])
              
                    with col1:
                        st.info("Text is below")
                        #Prints the text as the user provided
                        st.success(text_input)

                    with col2:
                        # Perform topic modeling on the transcript text
                        topics = perform_topic_modeling_LDA(text_input)
                        # Display the resulting topics in the app
                        st.info("Topics in the Text")
                        for topic in topics:
                            st.success(f"{topic[0]}: {', '.join(topic[1])}")

                    with col3:
                        st.info("Topic Labeling")
                        labeling_text = text_input
                        #Performs topic Labeling
                        industry = label_topic(labeling_text)
                        st.markdown("**Topic Labeling Industry Wise**")
                        st.write(industry)
            
    elif choice == "On Video":
        st.subheader("Topic Modeling and Labeling on Video")
       
        # Create a text input widget to allow users to enter a YouTube video URL
        url =  st.text_input('Enter URL of YouTube video:')

        if url is not None:
             
             if st.button("Analyze Video"):
                col1, col2, col3 = st.columns([1,1,1])

                with col1:
                    st.info("Video uploaded successfully")
                    #Downloads video to the desired location and also uploads on the screen
                    video_title, video_filename, video_filepath = save_video(url)
                    st.video(video_filepath)

                with col2:
                    st.info("Transcript is below") 
                    #Converts video into transcript
                    transcript_result = video_to_transcript(video_filename)
                    st.success(transcript_result)

                with col3:
                    st.info("Topic Modeling and Labeling")
                    labeling_text = transcript_result
                    #Performs Topic Labelling
                    industry = label_topic(labeling_text)
                    st.markdown("**Topic Labeling Industry Wise**")
                    st.write(industry)
                
    elif choice == "On CSV":
        st.subheader("Topic Modeling and Labeling on CSV File")
       
        # Create a file uploader widget to allow users to upload a CSV file
        upload_csv = st.file_uploader("Upload your CSV file", type=['csv'])

        if upload_csv is not None:

            if st.button("Analyze CSV File"):
                col1, col2 = st.columns([1,2])

                with col1:
                    st.info("CSV File uploaded")
                    csv_file = upload_csv.name
                   
                    # Opens the CSV file(if exists) as prints on the screen
                    with open(os.path.join(csv_file),"wb") as f: 
                        f.write(upload_csv.getbuffer()) 
                  
                    print(csv_file)
                    df = pd.read_csv(csv_file, encoding= 'unicode_escape')
                    st.dataframe(df)

                with col2:
                    data_list = df['Data'].tolist()
                    industry_list = []
                   
                    for i in data_list:
                        #Performs Topic Labelling
                        industry = label_topic(i)
                        industry_list.append(industry)
                  
                    df['Industry'] = industry_list
                    st.info("Topic Modeling and Labeling")
                    st.dataframe(df)

#Email Spam Classifier                    
elif option == "Email Spam Classifier":
	st.header("Enter the email you want to send")
	subject = st.text_input("Write the subject of the email", ' ')
	message = st.text_area("Add email Text Here", ' ')

	# Add button to check for spam 
	if st.button("Check"):
        # Create input 
		model_input = subject + ' ' + message
		# Process the data 
		model_input = sf.clean_text_spam(model_input)
		# Vectorize the inputs 
		vectorizer = joblib.load('Models/count_vectorizer_spam.sav')
		vec_inputs = vectorizer.transform(model_input)	
		# Load the model
		spam_model = joblib.load('Models/spam_model.sav')
		# Make the prediction 
		if spam_model.predict(vec_inputs):
			st.write("This message is **Spam**")
		else:
			st.write("This message is **Not Spam**")
               
# Keyword Sentiment Analysis
elif option == "Keyword Sentiment Analysis":
	st.header("Sentiment Analysis Tool")
	st.subheader("Enter the statement that you want to analyze")
	text_input = st.text_area("Enter sentence", height=50)
	# Model Selection 
	model_select = st.selectbox("Model Selection", ["Naive Bayes", "SVC", "Logistic Regression"])

	if st.button("Predict"):
		
		# Load the model 
		if model_select == "SVC":
			sentiment_model = joblib.load('Models/SVC_sentiment_model.sav')
		elif model_select == "Logistic Regression":
			sentiment_model = joblib.load('Models/LR_sentiment_model.sav')
		elif model_select == "Naive Bayes":
			sentiment_model = joblib.load('Models/NB_sentiment_model.sav')
		
		# Vectorize the inputs 
		vectorizer = joblib.load('Models/tfidf_vectorizer_sentiment_model.sav')
		vec_inputs = vectorizer.transform([text_input])
		# Keyword extraction 
		r = Rake(language='english')
		r.extract_keywords_from_text(text_input)
		# Get the important phrases
		phrases = r.get_ranked_phrases()
		
        # Make the prediction 
		if sentiment_model.predict(vec_inputs):
			st.write("This statement is **Positve**")
		else:
			st.write("This statement is **Negative**")

		# Display the important phrases
		st.write("These are the **keywords** causing the above sentiment:")
		for i, p in enumerate(phrases):
			st.write(i+1, p)

# Word Cloud Feature
elif option == "Word Cloud":
	st.header("Generate Word Cloud")
	st.subheader("Generate a word cloud from text containing the most popular words in the text.")
	# Ask for text or text file
	st.header('Enter text or upload file')
	text = st.text_area('Type Something', height=400)
	# Upload mask image 
	mask = st.file_uploader('Use Image Mask', type = ['jpg'])
	nltk.download('wordnet')
	# Add a button feature
	if st.button("Generate Wordcloud"):

		# Generate word cloud 
		st.write(len(text))
		nlp.create_wordcloud(text, mask)
		st.pyplot()

# N-Gram Analysis Option 
elif option == "N-Gram Analysis":
	st.header("N-Gram Analysis")
	st.subheader("This section displays the most commonly occuring N-Grams in your Data")
	# Ask for text or text file
	st.header('Enter text below')
	text = st.text_area('Type Something', height=400)
	# Parameters
	n = st.sidebar.slider("N for the N-gram", min_value=1, max_value=8, step=1, value=2)
	topk = st.sidebar.slider("Top k most common phrases", min_value=10, max_value=50, step=5, value=10)

	# Add a button 
	if st.button("Generate N-Gram Plot"): 
		# Plot the ngrams
		nlp.plot_ngrams(text, n=n, topk=topk)
		st.pyplot()

# Named Entity Recognition 
elif option == "Named Entity Recognition":
	st.header("Enter the statement that you want to analyze")
	st.markdown("**Random Sentence:** A Few Good Men is a 1992 American legal drama film set in Boston directed by Rob Reiner and starring Tom Cruise, Jack Nicholson, and Demi Moore. The film revolves around the court-martial of two U.S. Marines charged with the murder of a fellow Marine and the tribulations of their lawyers as they prepare a case to defend their clients.")
	text_input = st.text_area("Enter sentence")
	ner = en_core_web_sm.load()
	doc = ner(str(text_input))
	# Display 
	spacy_streamlit.visualize_ner(doc, labels=ner.get_pipe('ner').labels)

# Text Summarizer 
elif option == "Text Summarizer": 
	st.header("Text Summarization")
	st.subheader("Enter a corpus that you want to summarize")
	text_input = st.text_area("Enter a paragraph", height=150)
	sentence_count = len(sent_tokenize(text_input))
	st.write("Number of sentences:", sentence_count)
	model = st.sidebar.selectbox("Model Select", ["TextRank", "LexRank"])
	ratio = st.sidebar.slider("Select summary ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
	if st.button("Summarize"):
		if model == "TextRank":
			out = ts.text_sum_text(text_input, ratio=ratio)
		else:
			out = ts.text_sum_lex(text_input, ratio=ratio)
		st.write("**Summary Output:**", out)
		st.write("Number of output sentences:", len(sent_tokenize(out)))