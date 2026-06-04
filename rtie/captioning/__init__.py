"""Captioning: thumbnail image -> Ideogram-4 JSON caption (with bboxes).

Division of labor:
  - geometry (text boxes + exact strings via OCR, subject boxes via detector)
  - semantics (descriptions, style, archetype, emotion) via a vision model
  - assembly + validation against the Ideogram caption schema (CaptionVerifier)
"""
