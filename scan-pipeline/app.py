import asyncio
import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from pipeline import process_card

# converts BGR to RGB.
def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def to_bytes(image: np.ndarray | None) -> bytes | None:
    if image is None:
        return None
    return cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()

READERS = (("llm", "LLM"), ("trocr", "TrOCR"))

async def process_async(front: np.ndarray | None, back: np.ndarray | None, sources: dict):
    cards, errors = {}, {}
    for backend, label in READERS:
        try:
            cards[backend] = await process_card(to_bytes(front), to_bytes(back), sources, backend)
        except Exception as error:  
            errors[backend] = f"{type(error).__name__}: {error}"
    if not cards:
        raise gr.Error("; ".join(f"{label} failed: {errors[backend]}" for backend, label in READERS))

    base = next(iter(cards.values()))  
    images = []
    for side in ("front", "back"):
        result = base["sides"].get(side)
        images += [bgr_to_rgb(result["aligned"]), bgr_to_rgb(result["overlay"])] if result else [None, None]

    fields, scores = [], []
    for backend, label in READERS:
        if backend in cards:
            fields.append(json.dumps(cards[backend]["fields"], indent=2))
            scores.append(json.dumps(cards[backend]["score"], indent=2))
        else:
            fields.append(json.dumps({"error": errors[backend]}, indent=2))
            scores.append(json.dumps({"error": errors[backend]}, indent=2))
    return (*images, *fields, json.dumps(base["checkboxes"], indent=2), *scores)

# processes the card synchronously with process_async()
def process(front: np.ndarray | None, back: np.ndarray | None,
            front_source_file: str | None, back_source_file: str | None):
    if front is None and back is None:
        raise gr.Error("Upload a photo of the front, the back, or both.")
    sources = {}
    for side, photo, source_file in (("front", front, front_source_file), ("back", back, back_source_file)):
        if photo is None:
            continue
        if not source_file:
            raise gr.Error(f"Upload the {side} source JSON to go with the {side} photo.")
        try:
            sources[side] = json.loads(Path(source_file).read_text())
        except (OSError, ValueError) as error:
            raise gr.Error(f"Could not read the {side} source JSON: {error}")
    return asyncio.run(process_async(front, back, sources))

# creates the gradio interface.
with gr.Blocks(title="Scan Pipeline") as demo:
    gr.Markdown("Upload the front and back of a card, each with its source OCR JSON, to scan, align and score it. Fields are read by both the LLM and TrOCR so you can compare them. Either side can be left out, but a photo needs its source JSON.")
    with gr.Row():
        front = gr.Image(label="Front photo", type="numpy", sources=["upload", "webcam"])
        back = gr.Image(label="Back photo", type="numpy", sources=["upload", "webcam"])
    with gr.Row():
        front_source = gr.File(label="Front source OCR JSON", file_types=[".json"])
        back_source = gr.File(label="Back source OCR JSON", file_types=[".json"])
    process_btn = gr.Button("Process", variant="primary")
    with gr.Row():
        front_out = gr.Image(label="Front transformed")
        front_overlay = gr.Image(label="Front overlayed")
    with gr.Row():
        back_out = gr.Image(label="Back transformed")
        back_overlay = gr.Image(label="Back overlayed")
    with gr.Row():
        llm_fields_out = gr.Code(label="Extracted Fields (LLM)", language="json")
        trocr_fields_out = gr.Code(label="Extracted Fields (TrOCR)", language="json")
    with gr.Row():
        llm_score_out = gr.Code(label="Score (LLM)", language="json")
        trocr_score_out = gr.Code(label="Score (TrOCR)", language="json")
    checkboxes_out = gr.Code(label="Checkboxes", language="json")
    process_btn.click(
        process,
        [front, back, front_source, back_source],
        [front_out, front_overlay, back_out, back_overlay, llm_fields_out, trocr_fields_out,
         checkboxes_out, llm_score_out, trocr_score_out],
    )

if __name__ == "__main__":
    demo.launch(share=True)