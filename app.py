from flask import Flask, render_template, request
import requests
from translate import Translator

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def Index():
    return render_template("index.html")

@app.route("/Summarize", methods=["GET", "POST"])
def Summarize():
    if request.method == "POST":
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        headers = {"Authorization": f""}
        
        data = request.form["input"]
        maxL = int(request.form["max-len"])
        minL = maxL // 4
        input_lang = request.form["input-lang"]
        output_lang = request.form["output-lang"]

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()

        def chunk_text(text, chunk_size):
            words = text.split()
            for i in range(0, len(words), chunk_size):
                yield ' '.join(words[i:i + chunk_size])

        def translate_text(text, from_lang, to_lang):
            chunk_size = 50  # Adjust chunk size as needed
            chunks = list(chunk_text(text, chunk_size))
            translated_chunks = []
            translator = Translator(from_lang=from_lang, to_lang=to_lang)

            for chunk in chunks:
                try:
                    translated_chunks.append(translator.translate(chunk))
                except Exception as e:
                    translated_chunks.append(f"Error in translation: {str(e)}")
            
            return ' '.join(translated_chunks)

        # Translate input to English if necessary
        if input_lang != 'en':
            data = translate_text(data, input_lang, 'en')
        
        chunk_size = 500  # Divide into the chunk size as needed because our model has a word limit
        chunks = list(chunk_text(data, chunk_size))
        summaries = []
        
        for chunk in chunks:
            payload = {
                "inputs": chunk,
                "parameters": {"min_length": minL, "max_length": maxL}
            }
            try:
                output = query(payload)
                if len(output) > 0 and "summary_text" in output[0]:
                    summaries.append(output[0]["summary_text"])
                else:
                    summaries.append("Error: No summary returned for this chunk.")
            except Exception as e:
                summaries.append(f"An error occurred: {str(e)}")

        final_summary = ' '.join(summaries)
        
        # Translate output to the desired language if necessary
        if output_lang != 'en':
            final_summary = translate_text(final_summary, 'en', output_lang)

        return render_template("index.html", result=final_summary)

    return render_template("index.html")

if __name__ == '__main__':
    app.debug = True
    app.run()
