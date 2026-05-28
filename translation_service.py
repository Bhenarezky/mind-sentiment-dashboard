from deep_translator import GoogleTranslator

def translate_to_english(text):
    if not isinstance(text, str) or text.strip() == '':
        return ""
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated
    except Exception:
        return text