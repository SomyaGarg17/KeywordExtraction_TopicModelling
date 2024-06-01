#


# Import necessary libraries
from nltk.tokenize import sent_tokenize
import math
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Function to generate text summary using LexRank 
def text_sum_lex(text, ratio=0.2):
    # Create parse object 
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    
    total_sentence_count = len(sent_tokenize(text))
    gen_sentence_count = math.ceil(ratio * total_sentence_count)
    
    summarizer_lex = LexRankSummarizer()

    # Summarize using sumy LexRank
    summary = summarizer_lex(parser.document, gen_sentence_count)

    lex_summary = ""
    for sentence in summary:
        lex_summary = lex_summary + " " + str(sentence)
        
    return lex_summary

# Function to generate text summary using TextRank
def text_sum_text(text, ratio=0.2):
    # Create parse object 
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    
    total_sentence_count = len(sent_tokenize(text))
    gen_sentence_count = math.ceil(ratio * total_sentence_count)
    
    summarizer_text = TextRankSummarizer()

    # Summarize using sumy TextRank
    summary = summarizer_text(parser.document, gen_sentence_count)

    text_summary = ""
    for sentence in summary:
        text_summary = text_summary + " " + str(sentence)
        
    return text_summary
