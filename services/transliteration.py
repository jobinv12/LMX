from config import model

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

def indic_transliteration(src_lang:str, trgt_lang:str, text:str) -> str:

    src_lang_iso = transliteration_iso_codes[src_lang]
    trgt_lang_iso = transliteration_iso_codes[trgt_lang]

    messages = [
        (
            "system", f"You are a helpful assistant that transliterates {src_lang_iso} to {trgt_lang_iso}. Transliterate the user sentence."
        ),
        (
            "human", f"{text}"
        )
    ]

    response = model.invoke(messages)

    return response.content
