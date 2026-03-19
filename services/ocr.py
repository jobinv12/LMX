import os
from datetime import datetime
import shutil
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

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

model_id = "mlx-community/olmOCR-2-7B-1025-8bit"

try:
    # Load the model
    model, processor = load(model_id, trust_remote_code=True)
    config = load_config(model_id)
    print("olmocr2 model loaded successfully")
except Exception as e:
    print(f"Failed to load deepseekocr2 model {e}")

def upload_file(file):

    if file is None:
        return "No file uploaded"
    
    timestamp = datetime.now().strftime("%Y%m%d")

    original_name = os.path.basename(file.name)

    name, ext = os.path.splitext(original_name)

    new_filename = f"{name}_{timestamp}{ext}"

    dest_path = os.path.join("assets", new_filename)

    shutil.copy(file, dest_path)

    return dest_path, dest_path
    
def ocr(image_path:str, language:str) -> str:

    prompt = f"Convert the {language} document to markdown."
    
    formatted_output = apply_chat_template(processor, config, prompt, num_images=1)

    output = generate(model, processor, formatted_output, [image_path], verbose=False, max_tokens=2000)

    return output.text