import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


transliterate_languages: list = [
    "English",
    "Assamese",
    "Bengali",
    "Bodo",
    "Dogri",
    "Konkani",
    "Gujarati",
    "Hindi",
    "Kannada",
    "Kashmiri",
    "Kashmiri",
    "Maithili",
    "Malayalam",
    "Marathi",
    "Manipuri",
    "Manipuri",
    "Nepali",
    "Odia",
    "Punjabi",
    "Sanskrit",
    "Santali",
    "Sindhi",
    "Sindhi",
    "Tamil",
    "Telugu",
    "Urdu"
]

transliteration_iso_codes: dict = {
    "English": "eng_Latn",
    "Assamese": "asm_Beng",
    "Bengali": "ben_Beng",
    "Bodo": "brx_Deva",
    "Dogri": "doi_Deva",
    "Konkani": "gom_Deva",
    "Gujarati": "guj_Gujr",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Kashmiri (Arabic)": "kas_Arab",
    "Kashmiri (Devanagari)": "kas_Deva",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Marathi": "mar_Deva",
    "Manipuri (Bengali)": "mni_Beng",
    "Manipuri (Meitei)": "mni_Mtei",
    "Nepali": "npi_Deva",
    "Odia": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sindhi (Arabic)": "snd_Arab",
    "Sindhi (Devanagari) ()": "snd_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Urdu": "urd_Arab"
}

def indic_transliteration(model_name:str, src_lang:str, trgt_lang:str, text:str) -> str:

    model_id = model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id,trust_remote_code=True,torch_dtype=torch.bfloat16).to(DEVICE)
        print(f"{model_id} model loaded sucessfully.")
    except Exception as e:
        print(f"Failed to load {model_id} model.")

    ip = IndicProcessor(inference=True)

    batch = ip.preprocess_batch(
        [text],
        src_lang=src_lang,
        tgt_lang=trgt_lang
    )

    inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(DEVICE)

    with torch.no_grad():
        generated_tokens = model.generate(**inputs, use_cache=True, min_length=0, max_length=2048, num_beams=5, num_return_sequences=1)
    
    generated_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    translations = ip.postprocess_batch(generated_tokens, lang=trgt_lang)

    return "\n".join(translations)

def transliterate(src_lang:str, trgt_lang:str, text:str) -> str:

    src_lang_iso:str = transliteration_iso_codes.get(src_lang)
    trgt_lang_iso:str = transliteration_iso_codes.get(trgt_lang)

    if "eng_Latn" in src_lang_iso and "eng_Latn" not in trgt_lang_iso:
        return indic_transliteration("ai4bharat/indictrans2-en-indic-1B", src_lang_iso, trgt_lang_iso, text)
    elif "eng_Latn" not in src_lang_iso and "eng_Latn" in trgt_lang_iso:
        return indic_transliteration("ai4bharat/indictrans2-indic-en-1B", src_lang_iso, trgt_lang_iso, text)
    else:
        return indic_transliteration("ai4bharat/indictrans2-indic-indic-1B", src_lang_iso, trgt_lang_iso, text)
