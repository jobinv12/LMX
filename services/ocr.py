import os
from datetime import datetime
import shutil
from huggingface_hub import batch_bucket_files
from langchain_core.messages import HumanMessage
from config import model, HF_BUCKET_NAME, HF_BUCKET_URL, HF_BUCKET_TOKEN

ocr_languages = [
    "English", "Afrikaans", "Amharic", "Arabic", "Assamese", "Azerbaijani", 
    "Azerbaijani - Cyrillic", "Belarusian", "Bengali", "Tibetan", "Bosnian", 
    "Breton", "Bulgarian", "Catalan; Valencian", "Cebuano", "Czech", 
    "Chinese", "Cherokee", "Corsican", "Welsh", 
    "Danish", "Danish - Fraktur", "German", "German", 
    "Dzongkha", "Greek", "Esperanto", 
    "Estonian", "Basque", "Faroese", "Persian", "Filipino", 
    "Finnish", "French", "French", "Western Frisian", 
    "Scottish Gaelic", "Irish", "Galician", "Gujarati", 
    "Haitian; Haitian Creole", "Hebrew", "Hindi", "Croatian", "Hungarian", 
    "Armenian", "Inuktitut", "Indonesian", "Icelandic", "Italian",
    "Javanese", "Japanese", "Kannada", "Georgian", "Georgian - Old", "Kazakh", 
    "Central Khmer", "Kirghiz; Kyrgyz", "Kurmanji", "Korean", 
    "Korean", "Kurdish", "Lao", "Latin", "Latvian", 
    "Lithuanian", "Luxembourgish", "Malayalam", "Marathi", "Macedonian", 
    "Maltese", "Mongolian", "Maori", "Malay", "Burmese", "Nepali", 
    "Dutch; Flemish", "Norwegian", "Occitan", "Oriya", 
    "Panjabi; Punjabi", "Polish", "Portuguese", "Pushto; Pashto", "Quechua", 
    "Romanian; Moldavian; Moldovan", "Russian", "Sanskrit", "Sinhala; Sinhalese", 
    "Slovak", "Slovak - Fraktur", "Slovenian", "Sindhi", "Spanish; Castilian", 
    "Spanish; Castilian - Old", "Albanian", "Serbian", "Serbian - Latin", 
    "Sundanese", "Swahili", "Swedish", "Syriac", "Tamil", "Tatar", "Telugu", 
    "Tajik", "Thai", "Tigrinya", "Tonga", "Turkish", 
    "Uighur; Uyghur", "Ukrainian", "Urdu", "Uzbek", 
    "Vietnamese", "Yiddish", "Yoruba"
]

def upload_file(file):

    if file is None:
        return "No file uploaded"
    
    timestamp = datetime.now().strftime("%Y%m%d")

    original_name = os.path.basename(file.name)

    name, ext = os.path.splitext(original_name)

    new_filename = f"{name}_{timestamp}{ext}"

    os.makedirs("assets")

    dest_path = os.path.join("assets", new_filename)

    shutil.copy(file, dest_path)

    batch_bucket_files(
        bucket_id=HF_BUCKET_NAME,
        add=[
            (dest_path, new_filename)
        ],
        token=HF_BUCKET_TOKEN
    )

    file_url = f"{HF_BUCKET_URL}/{new_filename}"

    return dest_path, file_url
    
def ocr(image_url:str, language:str) -> str:
    
    messages = [
        HumanMessage(
       content=[
           {"type": "text", "text": "Extract the content of this document into well-formatted markdown. No Intro. Extract the user file. "},
           {
               "type": "image_url",
               "image_url": image_url
           }
       ]
    )
    ]

    response = model.invoke(messages)

    return response.content