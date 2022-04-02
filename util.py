import tensorflow as tf
import pickle
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# import os
# os.chdir('./server')

__classes = ['positive', 'negative']
# __model = None
# __tokenizer = None


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
                                "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


def cleanText(text):
    reg6 = re.compile(pattern=r'[^أ-ي]')
    text = reg6.sub(r' ', text)
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
    arb_stopwords = {'إي', 'شيكل', 'لبيك', 'خمس', 'والذين', 'هذه', 'هَذا', 'بكم', 'صراحة', 'حيَّ', 'ولكن', 'أخٌ', 'مكانكنّ', 'بعد', 'مكانكما', 'ة', 'بغتة', 'عاد', 'حتى', 'يناير', 'اللتان', 'عليه', 'اتخذ', 'خميس', 'حسب', 'تِي', 'ذانِ', 'ذال', 'إياي', 'رويدك', 'ط', 'بك', 'أفٍّ', 'ى', 'سنتيم', 'تاء', 'يمين', 'هم', 'بي', 'أف', 'طَق', 'لهما', 'ث', 'خاء', 'شمال', 'التي', 'فإذا', 'اخلولق', 'ديسمبر', 'لو', 'ليرة', 'بهن', 'ومن', 'ش', 'سقى', 'تينك', 'إنا', 'له', 'نيف', 'أصبح', 'به', 'ء', 'ع', 'كلَّا', 'كسا', 'سرعان', 'شين', 'حيث', 'تسعمئة', 'بات', 'لك', 'هَجْ', 'انقلب', 'وجد', 'قطّ', 'إليكن', 'ذِه', 'ن', 'هَاتِي', 'نحو', 'أوشك', 'استحال', 'إذا', 'فو', 'اللذين', 'صبر', 'خاصة', 'ثلاثاء', 'مارس', 'دال', 'مليم', 'غداة', 'صباح', 'ذهب', 'أمام', 'ذه', 'بما', 'سبعون', 'جيم', 'كلتا', 'ذو', 'كما', 'إياهما', 'خمسمائة', 'خبَّر', 'سوى', 'بمن', 'هاتين', 'ق', 'اللواتي', 'ارتدّ', 'أنتم', 'تلقاء', 'أوه', 'هبّ', 'عل', 'الآن', 'شبه', 'أيار', 'دينار', 'هَاتَيْنِ', 'ثمانية', 'تِه', 'حين', 'كاف', 'آهاً', 'غير', 'كلا', 'منذ', 'خلف', 'فرادى', 'أولالك', 'كيت', 'ظ', 'قلما', 'هذان', 'مرّة', 'بها', 'أمامك', 'لدى', 'تشرين', 'هذا', 'عشرة', 'لات', 'تحوّل', 'وإذا', 'واهاً', 'إليك', 'لستم', 'سبت', 'شتانَ', 'شتان', 'ما برح', 'ضاد', 'سادس', 'هاكَ', 'صدقا', 'لم', 'ذينك', 'الألاء', 'آه', 'لعل', 'تموز', 'ليسوا', 'اللائي', 'هنا', 'إيانا', 'كأيّن', 'مادام', 'انبرى', 'حاشا', 'بنا', 'ليست', 'يا', 'يفعلان', 'آذار', 'كرب', 'هاتان', 'كذلك', 'إلى', 'هن', 'كانون', 'اثني', 'تسع', 'ألا', 'نيسان', 'لكن', 'فاء', 'أسكن', 'هَيْهات', 'أخذ', 'لست', 'الألى', 'هَذِي', 'أيّ', 'ورد', 'ج', 'راء', 'حَذارِ', 'جوان', 'يوان', 'كأيّ', 'أكتوبر', 'بؤسا', 'منه', 'أضحى', 'يورو', 'آهٍ', 'رأى', 'علق', 'سحقا', 'ثالث', 'يوليو', 'مازال', 'أن', 'ذواتا', 'ز', 'مئتان', 'إى', 'عشرين', 'أين', 'سين', 'هل', 'ثمّة', 'ذلكم', 'زود', 'هلم', 'لولا', 'فبراير', 'كاد', 'ا', 'همزة', 'تين', 'أصلا', 'بماذا', 'أربعمائة', 'وإن', 'ستة', 'اربعين', 'هاتي', 'نحن', 'ذَيْنِ', 'ترك', 'رُبَّ', 'أمس', 'زعم', 'فيما', 'ظلّ', 'إنه', 'لما', 'كان', 'أم', 'جويلية', 'هَاتِه', 'يفعلون', 'طالما', 'بكن', 'فلس', 'عن', 'آمينَ', 'تَيْنِ', 'ذواتي', 'كلما', 'إيهٍ', 'اللتين', 'إذن', 'بسّ', 'ثمَّ', 'تلكم', 'مكانكم', 'بلى', 'تلك', 'أحد', 'ما', 'آض', 'نبَّا', 'تارة', 'عسى', 'بخٍ', 'صهْ', 'خامس', 'ريث', 'خمسون', 'مذ', 'طفق', 'ذلكن', 'لكم', 'رابع', 'ت', 'ي', 'مائة', 'لي', 'ذان', 'أيلول', 'ستين', 'أبريل', 'ثلاثة', 'ليستا', 'عشر', 'ك', 'آناء', 'لئن', 'غين', 'أنتن', 'فلان', 'شَتَّانَ', 'أيا', 'ءَ', 'فوق', 'أى', 'عوض', 'أنت', 'لام', 'مكانَك', 'خلافا', 'صبرا', 'لهم', 'إما', 'نعم', 'حجا', 'حمو', 'ثمانمئة', 'تسعين', 'ثمانون', 'مع', 'ته', 'م', 'أنّى', 'ذيت', 'معاذ', 'ذين', 'وُشْكَانَ', 'ثلاثمئة', 'عيانا', 'قاطبة', 'أخبر', 'ليسا', 'بئس', 'اللاتي', 'أربعة', 'أكثر', 'إياه', 'لكي', 'إليكم', 'ستون', 'هللة', 'درى', 'أمسى', 'مايو',
                     'مساء', 'ثمنمئة', 'إذ', 'مه', 'صار', 'ثمة', 'لكيلا', 'يونيو', 'هَذَيْنِ', 'أغسطس', 'لستن', 'ر', 'طاء', 'في', 'أعلم', 'واحد', 'إياكم', 'وهب', 'ليس', 'إزاء', 'هاته', 'مهما', 'قرش', 'لستما', 'أبٌ', 'نون', 'جمعة', 'إليكَ', 'نَخْ', 'كأي', 'فإن', 'أولئك', 'ذ', 'إياها', 'أجمع', 'لكنما', 'هيهات', 'ماذا', 'فيفري', 'بطآن', 'حقا', 'ألفى', 'راح', 'إياك', 'لا', 'ياء', 'خال', 'علًّ', 'كن', 'عين', 'تعسا', 'بيد', 'هاء', 'ذي', 'تعلَّم', 'ما أفعله', 'تحت', 'نوفمبر', 'لهن', 'أبدا', 'حار', 'لن', 'ثاء', 'هلا', 'كأنما', 'اربعون', 'خلا', 'مئة', 'ض', 'بضع', 'إلا', 'إياكما', 'عشرون', 'حبيب', 'نا', 'هناك', 'ص', 'أينما', 'هيا', 'وَيْ', 'سبعة', 'قام', 'ثم', 'أجل', 'بعدا', 'جنيه', 'والذي', 'وهو', 'أي', 'إلّا', 'كليهما', 'وإذ', 'درهم', 'د', 'أول', 'لمّا', 'بس', 'ين', 'ثلاثين', 'أما', 'هي', 'الذين', 'اللتيا', 'ستمائة', 'ولا', 'تلكما', 'اثنان', 'و', 'أيها', 'ولو', 'غالبا', 'ذلك', 'ثمانين', 'ستمئة', 'حرى', 'تفعلين', 'كليكما', 'أوّهْ', 'هاك', 'هَاتانِ', 'كى', 'اللذان', 'إن', 'أربعمئة', 'ذلكما', 'تخذ', 'أنتما', 'لا سيما', 'شرع', 'لسنا', 'ذا', 'أمامكَ', 'كِخ', 'اثنا', 'جلل', 'أنا', 'عَدَسْ', 'أربعاء', 'كلّما', 'زاي', 'هذين', 'سبعمائة', 'إذاً', 'على', 'لدن', 'أهلا', 'سوف', 'أقل', 'تي', 'لسن', 'سبعمئة', 'ثاني', 'أقبل', 'هؤلاء', 'رزق', 'جير', 'إنما', 'إمّا', 'فيه', 'هكذا', 'متى', 'تفعلون', 'بعض', 'إلَيْكَ', 'ابتدأ', 'أنشأ', 'عند', 'لكما', 'حمدا', 'هذي', 'عامة', 'جانفي', 'كأنّ', 'دونك', 'طاق', 'اثنين', 'مما', 'هنالك', 'ئ', 'إياكن', 'عاشر', 'ل', 'إياهم', 'أمد', 'مافتئ', 'خمسة', 'تانِك', 'ثمان', 'إيه', 'أ', 'حادي', 'أل', 'ساء', 'أنبأ', 'عجبا', 'بهما', 'هيّا', 'هيت', 'قاف', 'بل', 'هما', 'إليكنّ', 'حيثما', 'آ', 'فلا', 'ذات', 'غادر', 'ثان', 'إنَّ', 'ثلاثمائة', 'لكنَّ', 'تجاه', 'الذي', 'لوما', 'أيّان', 'حمٌ', 'هلّا', 'شباط', 'لها', 'ذانك', 'ضحوة', 'خ', 'لاسيما', 'فضلا', 'ؤ', 'خمسين', 'أخو', 'وما', 'بهم', 'كلاهما', 'كأن', 'هَذانِ', 'هَؤلاء', 'أيضا', 'طرا', 'أفريل', 'أُفٍّ', 'ظاء', 'حدَث', 'عما', 'دولار', 'نَّ', 'إياهن', 'ثلاثون', 'عدَّ', 'تسعمائة', 'سبعين', 'بكما', 'بين', 'ظنَّ', 'لعلَّ', 'آي', 'ريال', 'مثل', 'أنتِ', 'ثلاث', 'بَسْ', 'سبع', 'ما انفك', 'تانِ', 'أفعل به', 'أو', 'ليت', 'حاي', 'سرا', 'ذاك', 'منها', 'آها', 'ب', 'ميم', 'وا', 'ذِي', 'حبذا', 'تفعلان', 'عليك', 'ماي', 'واو', 'وراءَك', 'سمعا', 'تبدّل', 'تسعة', 'من', 'قد', 'فيم', 'بَلْهَ', 'كيف', 'أنًّ', 'سبحان', 'لعمر', 'تاسع', 'ثمّ', 'إليكما', 'جعل', 'خمسمئة', 'إذما', 'آهِ', 'أربع', 'جميع', 'دون', 'هَذِه', 'صاد', 'كم', 'آنفا', 'دواليك', 'كثيرا', 'كذا', 'سبتمبر', 'أنى', 'فيها', 'ذوا', 'كيفما', 'حزيران', 'أوت', 'تسعون', 'باء', 'ه', 'صهٍ', 'عدا', 'ثماني', 'هاهنا', 'ح', 'آب', 'ألف', 'غ', 'فمن', 'قبل', 'أعطى', 'أبو', 'نفس', 'أولاء', 'غدا', 'رجع', 'لنا', 'كأين', 'ست', 'كل', 'أرى', 'أمّا', 'هو', 'حاء', 'س', 'إحدى', 'كي', 'ها', 'ممن', 'سابع', 'ّأيّان', 'بخ', 'ثامن', 'علم', 'أطعم', 'ف'}
    important_stopWords = {'ألا', 'إلا', 'لا سيما', 'لا', 'لاسيما', 'لست',
                           'لعل', 'لكن', 'لم', 'لما', 'لن', 'ليس', 'ليست', 'ليستا', 'ولكن', 'لا'}
    arb_stopwords_updates = important_stopWords.symmetric_difference(
        arb_stopwords)
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
    # global __tokenizer
    # global __model
    try:
        __model = tf.keras.models.load_model('cnn_model.h5')
        with open('tokenizer.pickle', 'rb') as f:
            __tokenizer = pickle.load(f)
        text = deEmojify(text)
        text = cleanText(text)
        text = removeRepeatedCharacters(text)
        text = removeDiacritics(text)
        text = normalize_arabic(text)
        text = removeStopWords(text)
        text = stemming_processing(text)

        test = __tokenizer.texts_to_sequences([text])
        test = pad_sequences(test, maxlen=528)
        y_pred = __model.predict(test)

        return __classes[1 if y_pred >= 0.50 else 0]

    except Exception as e:
        print(e)


# def load_save_model():
#     global __model
#     try:
#         __model = tf.keras.models.load_model('cnn_model.h5')
#     except Exception as e:
#         print(e)


# if __name__ == '__main__':
#     # load_save_model()
#     # print(prediction(['الشقه ممتازه','الشقه مره سيئه']))
#     pass
