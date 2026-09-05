#!/usr/bin/env python
"""
Simple Gradio Web Application
Usage:
    python app.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath("src"))

from plant_disease.predict import predict


def create_app():
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is not installed. Install with: pip install gradio")
        return None

    def diagnose(image):
        if image is None:
            return "Please upload an image."

        res = predict(image)
        remedy = res['remedy']

        return f"""
### Disease: {remedy['name']} ({res['class_name']})
**Confidence:** {res['confidence']}%

- **Symptoms:** {remedy['symptoms']}
- **Treatment:** {remedy['treatment']}
- **Prevention:** {remedy['prevention']}
"""

    return gr.Interface(
        fn=diagnose,
        inputs=gr.Image(type="pil", label="Upload Leaf Image"),
        outputs=gr.Markdown(label="Diagnosis"),
        title="Plant Disease Predictor"
    )


if __name__ == "__main__":
    app = create_app()
    if app:
        app.launch()
