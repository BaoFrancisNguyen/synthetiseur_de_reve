import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from app import DreamSynthesizer, PromptManager

def test_prompt_manager_load_and_format():
    pm = PromptManager()
    prompt = pm.load_prompt("emotion_analysis")
    assert isinstance(prompt, str)
    assert "Analyse les émotions" in prompt

    formatted = pm.format_prompt("emotion_analysis", dream_text="Je vole dans le ciel.")
    assert "Je vole dans le ciel." in formatted

def test_dream_synthesizer_analyze_emotion(monkeypatch):
    ds = DreamSynthesizer()
    # Patch requests.post to simulate API response
    class DummyResponse:
        status_code = 200
        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": '{"heureux": 0.5, "stressant": 0.1, "neutre": 0.2, "triste": 0.1, "excitant": 0.1, "paisible": 0.0}'
                    }
                }]
            }
    monkeypatch.setattr("requests.post", lambda *a, **k: DummyResponse())
    emotions = ds.analyze_emotion("Un rêve de bonheur.")
    assert isinstance(emotions, dict)
    assert "heureux" in emotions

def test_generate_image_prompt(monkeypatch):
    ds = DreamSynthesizer()
    class DummyResponse:
        status_code = 200
        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": "A beautiful dreamlike scene with a princess."
                    }
                }]
            }
    monkeypatch.setattr("requests.post", lambda *a, **k: DummyResponse())
    prompt = ds.generate_image_prompt("Une princesse dans un château magique.")
    assert "princess" in prompt or "dreamlike" in prompt

def test_create_enhanced_placeholder_image():
    ds = DreamSynthesizer()
    img = ds.create_enhanced_placeholder_image("Un rêve de forêt magique.")
    assert img is not None
    assert hasattr(img, "save")  # PIL Image

def test_save_and_load_dream(tmp_path):
    ds = DreamSynthesizer()
    dream_data = {
        "id": "test1",
        "title": "Test Dream",
        "text": "Ceci est un rêve test.",
        "date": "2025-07-04T12:00:00",
        "emotions": {"heureux": 0.5}
    }
    # Patch Path to use tmp_path
    orig_path = ds.save_dream.__globals__["Path"]
    ds.save_dream.__globals__["Path"] = lambda x: tmp_path / x
    ds.load_dreams.__globals__["Path"] = lambda x: tmp_path / x
    try:
        assert ds.save_dream(dream_data) is True
        dreams = ds.load_dreams()
        assert isinstance(dreams, list)
        assert dreams[0]["title"] == "Test Dream"
    finally:
        ds.save_dream.__globals__["Path"] = orig_path
        ds.load_dreams.__globals__["Path"] = orig_path