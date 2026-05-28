import re
import string

# --- KAMUS PENDUKUNG PREPROCESSING ---
NEGATION_WORDS = {
    'not', 'no', 'never', 'neither', 'nor', 'none', 'cannot',
    'dont', 'doesnt', 'didnt', 'wont', 'wasnt', 'werent',
    'do not', 'does not', 'did not', 'will not', 'was not'
}

SLANG_DICT_ENG = {
    'u': 'you', 'r': 'are', 'idk': 'i do not know', 'omg': 'oh my god',
    'fav': 'favorite', 'thx': 'thanks', 'pls': 'please', 'plz': 'please',
    'lol': 'laughing out loud', 'lmao': 'laughing', 'wtf': 'what the heck',
    'imo': 'in my opinion', 'tbh': 'to be honest', 'ngl': 'not gonna lie',
    'cant': 'cannot', 'wont': 'will not', 'dont': 'do not',
    'didnt': 'did not', 'wasnt': 'was not', 'isnt': 'is not'
}

CONTRADICTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "won't": "will not", "wasn't": "was not", "weren't": "were not",
    "can't": "cannot", "couldn't": "could not", "shouldn't": "should not",
    "wouldn't": "would not", "isn't": "is not", "aren't": "are not",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "it's": "it is", "that's": "that is", "there's": "there is",
    "they're": "they are", "we're": "we are", "you're": "you are",
    "he's": "he is", "she's": "she is",
}

STOPWORDS_MINIMAL = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', 
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'or', 'if', 
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
    'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 
    'few', 'other', 'such', 'own', 'same', 'than', 'into'
}

def anonymized_tag_normalization(text):
    text = re.sub(r'\[NAME', '[NAME]', text)
    TAG_MAP = {
        r'\[NAME\]': 'someone',
        r'\[RELIGION\]': 'religion',
        r'\[TEAM\]': 'team',
        r'\[celebrity\]': 'famous_person',
        r'\[LOCATION\]': 'place'
    }
    for tag, replacement in TAG_MAP.items():
        text = re.sub(tag, replacement, text, flags=re.IGNORECASE)
    return text

def cleaning(text):
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    for contraction, expansion in CONTRADICTIONS.items():
        text = text.replace(contraction, expansion)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def slang_normalization(tokens):
    result = []
    for t in tokens:
        expanded = SLANG_DICT_ENG.get(t, t).split()
        result.extend(expanded)
    return result

def negation_handling(tokens):
    result = []
    i = 0
    while i < len(tokens):
        if tokens[i] in NEGATION_WORDS and i + 1 < len(tokens):
            result.append(tokens[i] + "_" + tokens[i+1])
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result

def full_preprocess(text):
    if not isinstance(text, str) or text.strip() == '':
        return ""

    text = anonymized_tag_normalization(text)
    text = text.lower()
    text = cleaning(text)

    tokens = text.split()
    tokens = slang_normalization(tokens)
    tokens = negation_handling(tokens)
    tokens = [t for t in tokens if t not in STOPWORDS_MINIMAL]
    
    # PERBAIKAN TOTAL: Menggunakan Lemmatizer Kamus Manual yang Stabil (Bebas dari Bug NameError Word)
    lema_map = {
        'tired': 'tire', 'exhausted': 'exhaust', 'overwhelmed': 'overwhelm', 
        'annoyed': 'annoy', 'disappointed': 'disappoint', 'happy': 'happi',
        'feeling': 'feel', 'feels': 'feel'
    }
    tokens = [lema_map.get(t, t) for t in tokens]

    result = ' '.join(tokens)
    return result if result.strip() else 'neutral'