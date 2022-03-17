import tensorflow as tf
import pickle
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences

__classes = ['positive','negative']
__model = None

def deEmojify(text):
    regrex_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            u"\U0001f926-\U0001f937"
                            u"\U00010000-\U0010ffff"
                            u"\u2640-\u2642"
                            u"\u2600-\u2B55"
                            u"\u200d"
                            u"\u23cf"
                            u"\u23e9"
                            u"\u231a"
                            u"\ufe0f"
                            # dingbats u "\u3030"
                            "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def cleanText(text):
    reg6 = re.compile(pattern = r'[^أ-ي]')
    text = reg6.sub(r' ',text)
    return text

def removeRepeatedCharacters(text):
    t = ' '
    list_tokens = []
    for i in text.split(' '):
        list_tokens.append(str(re.sub(r'(.)\1+', r"\1\1", i)))
    t = t.join(tuple(list_tokens))
    return t

def removeDiacritics(text):
    arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    text = arabic_diacritics.sub('', text)
    return text

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

def removeStopWords(text):
    t = ' '
    arb_stopwords = set(nltk.corpus.stopwords.words("arabic"))
    important_stopWords = {'ألا','إلا','لا سيما','لا','لاسيما','لست','لعل','لكن','لم','لما','لن','ليس','ليست','ليستا','ولكن','لا'}
    arb_stopwords_updates = important_stopWords.symmetric_difference(arb_stopwords)
    text = word_tokenize(text)
     
    #filtered_sentence = [w for w in text if not w.lower() in arb_stopwords_updates]
 
    filtered_sentence = []
 
    for w in text:
        if w not in arb_stopwords_updates:
            filtered_sentence.append(w)
    t = t.join(filtered_sentence)
    return t


def stemming_processing(text):
    stem = PorterStemmer()
    tokens = word_tokenize(text)
    sents = ' '
    listOfTokens = []
    for token in tokens:
        listOfTokens.append(stem.stem(token))
    
    sents = sents.join(tuple(listOfTokens))
    return sents





def prediction(text):

    try:
        with open(r'D:\Files\Projects\NLP\Arabic sentiment analysis\server\emotions\tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
        text = deEmojify(text)
        text = cleanText(text)
        text = removeRepeatedCharacters(text)
        text = removeDiacritics(text)
        text = normalize_arabic(text)
        text = removeStopWords(text)
        text = stemming_processing(text)

        test = tokenizer.texts_to_sequences([text])
        test = pad_sequences(test, maxlen=528)
        y_pred = __model.predict(test)

        return __classes[1 if y_pred >= 0.50 else 0]

    except Exception as e:
        pass

def load_save_model():
    global __model

    __model = tf.keras.models.load_model(r'D:\Files\Projects\NLP\Arabic sentiment analysis\server\emotions\cnn_model.h5')


if __name__ == '__main__':
    # load_save_model()
    # print(prediction(['الشقه ممتازه','الشقه مره سيئه']))
    pass
