import gradio as gr
from services.ocr import upload_file, ocr, ocr_languages
from services.interleave import interleave
from services.translation import translation_languages, translation
from services.transliteration import transliterate_languages, transliterate

img_path="bmc/qr-code.png"

html_header = f"""
<div style="display:flex; align-items:center; justify-content:space-between;">
    <h1 style="margin:0;">LMX</h1>
</div>
"""

html_footer = f"""
<footer>
</footer>
"""

with gr.Blocks(title="LMX", analytics_enabled=True) as interface:

    with gr.Row():
        gr.HTML(html_header)
        gr.Button(variant="primary", scale=1, min_width=5, value="☕️ Buy me a coffee", size="lg", link_target="_blank", link="https://buymeacoffee.com/jobvargh")
        gr.Button(variant="primary", scale=1, min_width=5, value="Github", size="lg", link_target="_blank", link="https://github.com/jobinv12/LMX")

    
    with gr.Tab("OCR"):
        with gr.Row():
            with gr.Column():
                ocr_path = gr.Textbox(interactive=False, visible=False)
                ocr_languages = gr.Dropdown(label="Source language", choices=ocr_languages, filterable=True, info="Select source language")
                ocr_upload_file = gr.UploadButton(label="Upload a Image", file_types=[".png",".jpeg",".webp",".jpg"], file_count="single")
                ocr_file = gr.Image(height=250, width=700, type="filepath", show_label=False, buttons=[], container=False)
                ocr_upload_file.upload(upload_file, inputs=ocr_upload_file, outputs=[ocr_file, ocr_path])

            with gr.Column():
                ocr_output = gr.TextArea(label="Output", lines=15 ,interactive=False)

        with gr.Row():
            clear_btn = gr.ClearButton(value="Clear", variant="secondary")
            ocr_submit_btn = gr.Button(value="Run", variant="huggingface")

        ocr_submit_btn.click(fn=ocr, inputs=[ocr_path, ocr_languages], outputs=ocr_output)

    with gr.Tab("Transliteration"):
        with gr.Row():
            with gr.Column():
                transliteration_src_lang = gr.Dropdown(label="Source Language", choices=transliterate_languages, filterable=True, interactive=True)
                transliteration_input_area = gr.TextArea(label="Input")

            with gr.Column():
                transliteration_trgt_lang = gr.Dropdown(label="Target Language", choices=transliterate_languages, filterable=True, interactive=True)
                transliteration_output_area = gr.TextArea(label="Output", interactive=False)
    
        with gr.Row():
            clear_btn = gr.ClearButton(value="Clear", variant="secondary")
            transliteration_submit_btn = gr.Button(value="Run", variant="huggingface")
        
        transliteration_submit_btn.click(fn=transliterate, inputs=[transliteration_src_lang, transliteration_trgt_lang, transliteration_input_area], outputs=transliteration_output_area)
    
    with gr.Tab("Translation"):
        with gr.Row():
            with gr.Column():
                translation_src_lang = gr.Dropdown(label="Source Language", choices=translation_languages, filterable=True, interactive=True)
                translation_input_area = gr.TextArea(label="Input")
            with gr.Column():
                translation_trgt_lang = gr.Dropdown(label="Target Language", choices=translation_languages, filterable=True, interactive=True)
                translation_output_area = gr.TextArea(label="Output", interactive=False)
    
        with gr.Row():
            clear_btn = gr.ClearButton(value="Clear", variant="secondary")
            translation_submit_btn = gr.Button(value="Run", variant="huggingface")  

        translation_submit_btn.click(fn=translation, inputs=[translation_src_lang, translation_trgt_lang, translation_input_area], outputs=translation_output_area) 


    with gr.Tab("Interleave"):
        with gr.Row():
            interleave_input_area = gr.TextArea(label="First Input", type="text")
            interleave_input2_area = gr.TextArea(label="Second Input", type="text")

        with gr.Row():
            clear_btn = gr.ClearButton(value="Clear", variant="secondary")
            interleave_submit_btn = gr.Button(value="Run" ,variant="huggingface")    

        with gr.Row():
            interleave_output_area = gr.TextArea(label="Output", interactive=False, type="text")
        interleave_submit_btn.click(fn=interleave, inputs=[interleave_input_area, interleave_input2_area], outputs=[interleave_output_area])
    
    with gr.Row():
        gr.HTML(html_template=html_footer)

if __name__ == "__main__":
    interface.launch(pwa=True, share=True, allowed_paths=["assets", "bmc"])