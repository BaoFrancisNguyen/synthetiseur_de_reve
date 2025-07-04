import streamlit as st
import speech_recognition as sr
import requests
import json
import base64
import io
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from PIL import Image
import os
from pathlib import Path
import yaml
import random
import time

# Configuration du logging pour le débogage
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class PromptManager:
    """
    Gestionnaire des prompts pour les différentes APIs.
    Charge les prompts depuis des fichiers séparés pour une meilleure maintenance.
    """
    
    def __init__(self):
        """Initialise le gestionnaire de prompts."""
        logger.debug("Initialisation du PromptManager")
        self.prompts_dir = Path("prompts")
        self.prompts_dir.mkdir(exist_ok=True)
        self._create_default_prompts()
        
    def _create_default_prompts(self):
        """Crée les fichiers de prompts par défaut s'ils n'existent pas."""
        logger.debug("Création des prompts par défaut")
        
        # Prompt pour l'analyse émotionnelle
        emotion_prompt = """
        Analyse les émotions dans ce rêve et retourne uniquement un JSON avec les scores de 0 à 1.
        
        Rêve: {dream_text}
        
        Consignes:
        - Analyse le contenu émotionnel du rêve
        - Évalue l'intensité de chaque émotion de 0 (absent) à 1 (très intense)
        - Les scores doivent être cohérents avec le contenu du rêve
        - Retourne uniquement le JSON, sans explication
        
        Format de réponse attendu:
        {{
            "heureux": 0.0,
            "stressant": 0.0,
            "neutre": 0.0,
            "triste": 0.0,
            "excitant": 0.0,
            "paisible": 0.0
        }}
        """
        
        # Prompt pour la génération d'image
        image_prompt = """
        Tu es un expert en génération de prompts pour des images de rêves oniriques.
        
        Consignes:
        - Transforme ce rêve en un prompt descriptif et artistique
        - Le prompt doit être en anglais
        - Utilise un style visuel onirique et surréaliste
        - Inclus des détails visuels spécifiques (couleurs, lumière, atmosphère)
        - Reste fidèle au contenu émotionnel du rêve
        - Maximum 150 mots
        
        Rêve à transformer: {dream_text}
        
        Commence directement par le prompt en anglais, sans introduction.
        """
        
        # Sauvegarde des prompts
        prompts = {
            "emotion_analysis": emotion_prompt,
            "image_generation": image_prompt
        }
        
        for prompt_name, prompt_content in prompts.items():
            prompt_file = self.prompts_dir / f"{prompt_name}.txt"
            if not prompt_file.exists():
                with open(prompt_file, "w", encoding="utf-8") as f:
                    f.write(prompt_content)
                logger.debug(f"Prompt {prompt_name} créé")
    
    def load_prompt(self, prompt_name: str) -> str:
        """
        Charge un prompt depuis un fichier.
        
        Args:
            prompt_name: Nom du prompt à charger
            
        Returns:
            str: Contenu du prompt
        """
        logger.debug(f"Chargement du prompt: {prompt_name}")
        
        prompt_file = self.prompts_dir / f"{prompt_name}.txt"
        
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                content = f.read()
                logger.debug(f"Prompt {prompt_name} chargé avec succès")
                return content
        except FileNotFoundError:
            logger.error(f"Fichier de prompt non trouvé: {prompt_file}")
            return ""
        except Exception as e:
            logger.error(f"Erreur lors du chargement du prompt {prompt_name}: {e}")
            return ""
    
    def format_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Charge et formate un prompt avec les variables fournies.
        
        Args:
            prompt_name: Nom du prompt
            **kwargs: Variables pour le formatage
            
        Returns:
            str: Prompt formaté
        """
        logger.debug(f"Formatage du prompt {prompt_name} avec {kwargs.keys()}")
        
        prompt_template = self.load_prompt(prompt_name)
        
        try:
            formatted_prompt = prompt_template.format(**kwargs)
            logger.debug(f"Prompt formaté avec succès")
            return formatted_prompt
        except KeyError as e:
            logger.error(f"Variable manquante pour le prompt {prompt_name}: {e}")
            return prompt_template
        except Exception as e:
            logger.error(f"Erreur lors du formatage du prompt {prompt_name}: {e}")
            return prompt_template
    

    
class DreamSynthesizer:
    """
    Classe principale pour le synthétiseur de rêves avec génération d'images corrigée.
    """
    
    def __init__(self):
        """Initialise le synthétiseur avec les configurations nécessaires."""
        logger.debug("Initialisation du DreamSynthesizer")
        
        # Configuration des APIs
        self.groq_api_key = st.secrets.get("GROQ_API_KEY", "")
        self.clipdrop_api_key = st.secrets.get("CLIPDROP_API_KEY", "")
        self.mistral_api_key = st.secrets.get("MISTRAL_API_KEY", "")
        self.hf_token = st.secrets.get("HF_TOKEN", "")
        self.openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
        self.replicate_token = st.secrets.get("REPLICATE_API_TOKEN", "")
        
        # URLs des APIs
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.mistral_url = "https://api.mistral.ai/v1/chat/completions"
        
        # Initialisation du recognizer pour la reconnaissance vocale
        self.recognizer = sr.Recognizer()
        
        # Initialisation du gestionnaire de prompts
        self.prompt_manager = PromptManager()
        
        logger.debug("DreamSynthesizer initialisé avec succès")

    def transcribe_audio(self, audio_file) -> str:
        """
        Transcrit un fichier audio en texte avec Groq Whisper (gratuit) + fallback Google.
        """
        logger.debug("🎤 Début de la transcription audio")
        
        # Méthode 1: Groq Whisper (gratuit, rapide)
        if self.groq_api_key:
            try:
                logger.debug("🔄 Tentative avec Groq Whisper")
                
                headers = {
                    "Authorization": f"Bearer {self.groq_api_key}",
                }
                
                # Préparer le fichier pour l'API
                audio_file.seek(0)
                files = {
                    "file": ("audio.wav", audio_file.getvalue(), "audio/wav"),
                    "model": (None, "whisper-large-v3"),
                    "language": (None, "fr"),
                    "response_format": (None, "text")
                }
                
                response = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    files=files,
                    timeout=30
                )
                
                if response.status_code == 200:
                    text = response.text.strip()
                    logger.info(f"✅ Transcription Groq réussie: {len(text)} caractères")
                    return text
                else:
                    logger.warning(f"⚠️ Groq Whisper échec: {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Erreur Groq Whisper: {e}")
        
        # Méthode 2: Fallback vers Google Speech Recognition
        try:
            logger.debug("🔄 Fallback vers Google Speech Recognition")
            
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_file.seek(0)
                temp_file.write(audio_file.getbuffer())
                temp_path = temp_file.name
            
            with sr.AudioFile(temp_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language="fr-FR")
                
                logger.info(f"✅ Transcription Google réussie: {len(text)} caractères")
                return text
                
        except Exception as e:
            error_msg = f"Erreur lors de la transcription: {e}"
            logger.error(error_msg)
            return error_msg
        finally:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)

    def analyze_emotion(self, dream_text: str) -> Dict[str, float]:
        """
        Analyse l'émotion d'un rêve en utilisant l'API Mistral.
        """
        logger.debug(f"Début de l'analyse émotionnelle pour: {dream_text[:50]}...")
        
        # Émotions par défaut en cas d'erreur
        default_emotions = {
            "heureux": 0.3,
            "stressant": 0.2,
            "neutre": 0.4,
            "triste": 0.1,
            "excitant": 0.0,
            "paisible": 0.0
        }
        
        try:
            # Chargement du prompt depuis le fichier
            prompt = self.prompt_manager.format_prompt("emotion_analysis", dream_text=dream_text)
            
            # Requête à l'API Mistral
            headers = {
                "Authorization": f"Bearer {self.mistral_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "mistral-tiny",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200
            }
            
            logger.debug("Envoi de la requête à l'API Mistral")
            response = requests.post(self.mistral_url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                emotion_text = result['choices'][0]['message']['content']
                logger.debug(f"Réponse de l'API: {emotion_text}")
                
                # Parsing du JSON retourné
                try:
                    emotions = json.loads(emotion_text)
                    logger.debug(f"Émotions analysées: {emotions}")
                    return emotions
                except json.JSONDecodeError:
                    logger.warning("Impossible de parser le JSON des émotions")
                    return default_emotions
            else:
                logger.error(f"Erreur API Mistral: {response.status_code}")
                return default_emotions
                
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse émotionnelle: {e}")
            return default_emotions

    def generate_image_prompt(self, dream_text: str) -> str:
        """
        Génère un prompt optimisé pour la génération d'image.
        """
        logger.debug(f"Génération du prompt pour: {dream_text[:50]}...")
        
        try:
            # Chargement du prompt depuis le fichier
            prompt = self.prompt_manager.format_prompt("image_generation", dream_text=dream_text)
            
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 150,
                "temperature": 0.7
            }
            
            logger.debug("Envoi de la requête à l'API Groq pour le prompt")
            response = requests.post(self.groq_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                image_prompt = result['choices'][0]['message']['content'].strip()
                
                # Nettoyage du prompt
                image_prompt = image_prompt.replace("Here is the optimized prompt:", "").strip()
                image_prompt = image_prompt.replace('"', '').strip()
                
                logger.debug(f"Prompt généré: {image_prompt}")
                return image_prompt
            else:
                logger.error(f"Erreur API Groq: {response.status_code}")
                return self._create_fallback_prompt(dream_text)
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération du prompt: {e}")
            return self._create_fallback_prompt(dream_text)

    def _create_fallback_prompt(self, dream_text: str) -> str:
        """Crée un prompt de fallback basé sur des mots-clés."""
        keywords = {
            'princess': 'beautiful princess in ethereal light',
            'castle': 'majestic castle with magical atmosphere',
            'forest': 'mystical forest with glowing trees',
            'ocean': 'vast ocean with shimmering waves',
            'flying': 'floating weightlessly through clouds',
            'magic': 'magical sparkles and glowing aura',
            'night': 'serene nighttime with starry sky',
            'colors': 'vibrant rainbow colors swirling',
            'fear': 'dark shadows with mysterious atmosphere',
            'love': 'warm golden light with soft glow'
        }
        
        prompt_parts = ["Dreamlike surreal scene"]
        text_lower = dream_text.lower()
        
        for keyword, description in keywords.items():
            if keyword in text_lower:
                prompt_parts.append(description)
        
        prompt_parts.extend([
            "digital art style",
            "soft lighting",
            "magical atmosphere",
            "high quality",
            "detailed"
        ])
        
        return ", ".join(prompt_parts)
    
    def generate_video_prompt(self, dream_text: str) -> str:
        """
        Génère un prompt optimisé pour la génération de vidéo.
        
        Args:
            dream_text (str): Texte du rêve
            
        Returns:
            str: Prompt optimisé pour la vidéo
        """
        logger.debug(f"Génération du prompt vidéo pour: {dream_text[:50]}...")
        
        try:
            # Prompt spécialisé pour la vidéo
            video_prompt_template = """
            Tu es un expert en génération de prompts pour des vidéos oniriques et surréalistes.
            
            Consignes:
            - Transforme ce rêve en un prompt descriptif pour une vidéo courte
            - Le prompt doit être en anglais
            - Utilise un style visuel cinématographique et onirique
            - Inclus des éléments de mouvement et d'animation (floating, flowing, shifting)
            - Reste fidèle au contenu émotionnel du rêve
            - Mentionne des effets visuels appropriés (particles, light rays, morphing)
            - Maximum 100 mots
            
            Rêve à transformer: {dream_text}
            
            Commence directement par le prompt en anglais, sans introduction.
            """
            
            prompt = video_prompt_template.format(dream_text=dream_text)
            
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 120,
                "temperature": 0.8
            }
            
            logger.debug("Envoi de la requête à l'API Groq pour le prompt vidéo")
            response = requests.post(self.groq_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                video_prompt = result['choices'][0]['message']['content'].strip()
                
                # Nettoyage du prompt
                video_prompt = video_prompt.replace("Here is the optimized prompt:", "").strip()
                video_prompt = video_prompt.replace('"', '').strip()
                
                logger.debug(f"Prompt vidéo généré: {video_prompt}")
                return video_prompt
            else:
                logger.error(f"Erreur API Groq pour prompt vidéo: {response.status_code}")
                return self._create_fallback_video_prompt(dream_text)
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération du prompt vidéo: {e}")
            return self._create_fallback_video_prompt(dream_text)

    def generate_music_prompt(self, dream_text: str) -> str:
        """
        Génère un prompt optimisé pour la génération de musique.
        
        Args:
            dream_text (str): Texte du rêve
            
        Returns:
            str: Prompt optimisé pour la musique
        """
        logger.debug(f"Génération du prompt musical pour: {dream_text[:50]}...")
        
        try:
            # Prompt spécialisé pour la musique
            music_prompt_template = """
            Tu es un expert en génération de prompts pour de la musique ambiante et onirique.
            
            Consignes:
            - Transforme ce rêve en un prompt descriptif pour de la musique d'ambiance
            - Le prompt doit être en anglais
            - Utilise des termes musicaux appropriés (ambient, ethereal, dreamy, mystical)
            - Inclus des instruments et textures sonores (synth pads, piano, strings, nature sounds)
            - Reste fidèle à l'atmosphère émotionnelle du rêve
            - Mentionne le tempo et l'ambiance générale
            - Maximum 80 mots
            
            Rêve à transformer: {dream_text}
            
            Commence directement par le prompt musical en anglais, sans introduction.
            """
            
            prompt = music_prompt_template.format(dream_text=dream_text)
            
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            logger.debug("Envoi de la requête à l'API Groq pour le prompt musical")
            response = requests.post(self.groq_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                music_prompt = result['choices'][0]['message']['content'].strip()
                
                # Nettoyage du prompt
                music_prompt = music_prompt.replace("Here is the optimized prompt:", "").strip()
                music_prompt = music_prompt.replace('"', '').strip()
                
                logger.debug(f"Prompt musical généré: {music_prompt}")
                return music_prompt
            else:
                logger.error(f"Erreur API Groq pour prompt musical: {response.status_code}")
                return self._create_fallback_music_prompt(dream_text)
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération du prompt musical: {e}")
            return self._create_fallback_music_prompt(dream_text)

    def _create_fallback_video_prompt(self, dream_text: str) -> str:
        """Crée un prompt vidéo de fallback basé sur des mots-clés."""
        keywords = {
            'princess': 'ethereal princess floating gracefully through magical light',
            'castle': 'majestic castle with slowly shifting magical architecture',
            'forest': 'mystical forest with gently swaying glowing trees',
            'ocean': 'vast ocean with mesmerizing wave movements and light reflections',
            'flying': 'smooth flying movement through dreamy cloudscapes',
            'magic': 'magical particles flowing and sparkling in slow motion',
            'night': 'serene nighttime with twinkling stars and gentle moon glow',
            'colors': 'vibrant colors morphing and blending seamlessly',
            'dancing': 'graceful dancing movements with flowing fabric',
            'fear': 'dark atmospheric shadows with mysterious movements'
        }
        
        prompt_parts = ["Cinematic dreamlike video sequence"]
        text_lower = dream_text.lower()
        
        for keyword, description in keywords.items():
            if keyword in text_lower:
                prompt_parts.append(description)
        
        prompt_parts.extend([
            "slow motion effects",
            "soft ethereal lighting",
            "magical atmosphere",
            "smooth camera movements",
            "particle effects",
            "high quality animation"
        ])
        
        return ", ".join(prompt_parts)

    def _create_fallback_music_prompt(self, dream_text: str) -> str:
        """Crée un prompt musical de fallback basé sur des mots-clés."""
        keywords = {
            'princess': 'ethereal ambient music with delicate piano and soft strings',
            'castle': 'majestic orchestral ambient with gentle horn melodies',
            'forest': 'nature ambient with subtle woodland sounds and mystical pads',
            'ocean': 'flowing ambient with water sounds and gentle waves',
            'peaceful': 'serene ambient meditation music with soft drones',
            'magic': 'mystical ambient with sparkling bell tones and ethereal voices',
            'night': 'nocturnal ambient with gentle synth pads and distant echoes',
            'fear': 'dark ambient with deep drones and mysterious textures',
            'happy': 'uplifting ambient with warm tones and gentle melodies',
            'sad': 'melancholic ambient with minor key piano and strings'
        }
        
        prompt_parts = ["Dreamy ambient soundscape"]
        text_lower = dream_text.lower()
        
        for keyword, description in keywords.items():
            if keyword in text_lower:
                prompt_parts.append(description)
                break
        
        prompt_parts.extend([
            "slow tempo",
            "ethereal atmosphere",
            "soft reverb",
            "meditation style",
            "10 seconds duration"
        ])
        
        return ", ".join(prompt_parts)



    def generate_image(self, dream_text: str) -> Tuple[Optional[Image.Image], str]:
        """
        Génère une image à partir du texte du rêve avec fallbacks multiples - VERSION AMÉLIORÉE.
        Retourne l'image générée et le prompt utilisé.
        """
        logger.debug(f"🖼️ Génération d'image pour: {dream_text[:50]}...")
        
        # Génération du prompt optimisé
        image_prompt = self.generate_image_prompt(dream_text)
        logger.info(f"Prompt utilisé: {image_prompt}")
        
        # Liste des méthodes par ordre de priorité (les plus fiables en premier)
        methods = [
            ("Pollinations AI (Gratuit)", self._generate_pollinations),  # 100% gratuit, très fiable
            ("Prodia (Gratuit)", self._generate_prodia),                # Gratuit avec compte
            ("Runware", self._generate_runware),                        # Payant mais très rapide
            ("StableDiffusionAPI", self._generate_stablediffusionapi),  # API spécialisée
            ("Hugging Face", self._generate_huggingface),              # Gratuit mais parfois lent
            ("ClipDrop", self._generate_clipdrop),                     # Payant
            ("OpenAI DALL-E", self._generate_openai),                  # Payant, haute qualité
            ("Replicate", self._generate_replicate),                   # Payant
            ("Local Stable Diffusion", self._generate_local_diffusion), # Local
        ]
        
        for method_name, method_func in methods:
            try:
                logger.debug(f"📡 Tentative {method_name}")
                image = method_func(image_prompt)
                
                if image:
                    logger.info(f"✅ Image générée avec succès via {method_name}")
                    return image, image_prompt
                else:
                    logger.warning(f"⚠️ {method_name} n'a pas produit d'image")
                    
            except Exception as e:
                logger.error(f"❌ Erreur {method_name}: {e}")
                continue
        
        # Fallback final : placeholder amélioré
        logger.debug("🎨 Fallback final : génération d'une image placeholder améliorée")
        placeholder_image = self.create_enhanced_placeholder_image(dream_text)
        return placeholder_image, image_prompt

    def _generate_clipdrop(self, prompt: str) -> Optional[Image.Image]:
        """Génère une image avec ClipDrop."""
        if not self.clipdrop_api_key:
            return None
        
        try:
            headers = {
                "x-api-key": self.clipdrop_api_key,
            }
            
            # ClipDrop utilise form-data
            data = {
                "prompt": prompt[:500],  # Limitation de longueur
                "width": "512",
                "height": "512"
            }
            
            response = requests.post(
                "https://clipdrop-api.co/text-to-image/v1",
                headers=headers,
                data=data,
                timeout=120
            )
            
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                return image
            else:
                logger.error(f"ClipDrop error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"ClipDrop exception: {e}")
            return None

    def _generate_huggingface(self, prompt: str) -> Optional[Image.Image]:
        """Génère une image avec Hugging Face Stable Diffusion."""
        if not self.hf_token:
            return None
        
        try:
            # Utilisation d'un modèle Stable Diffusion plus récent et fiable
            model_url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
            
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "negative_prompt": "blurry, low quality, distorted, ugly, deformed, text, watermark",
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5,
                    "width": 512,
                    "height": 512,
                    "seed": -1
                },
                "options": {
                    "wait_for_model": True,
                    "use_cache": False
                }
            }
            
            # Première tentative
            response = requests.post(model_url, headers=headers, json=payload, timeout=300)
            
            if response.status_code == 503:
                # Modèle en chargement, attendre et réessayer
                logger.debug("Modèle en chargement, attente de 20 secondes...")
                time.sleep(20)
                response = requests.post(model_url, headers=headers, json=payload, timeout=300)
            
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                return image
            else:
                logger.error(f"HuggingFace error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"HuggingFace exception: {e}")
            return None

    def _generate_openai(self, prompt: str) -> Optional[Image.Image]:
        """Génère une image avec OpenAI DALL-E."""
        if not self.openai_api_key:
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "dall-e-3",
                "prompt": prompt[:4000],  # DALL-E a une limite de caractères
                "n": 1,
                "size": "1024x1024",
                "quality": "standard",
                "style": "vivid"
            }
            
            response = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                image_url = result['data'][0]['url']
                
                # Télécharger l'image
                img_response = requests.get(image_url, timeout=60)
                if img_response.status_code == 200:
                    # Redimensionner à 512x512 pour cohérence
                    image = Image.open(io.BytesIO(img_response.content))
                    image = image.resize((512, 512), Image.Resampling.LANCZOS)
                    return image
            else:
                logger.error(f"OpenAI error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"OpenAI exception: {e}")
            return None

    def _generate_replicate(self, prompt: str) -> Optional[Image.Image]:
        """Génère une image avec Replicate."""
        if not self.replicate_token:
            return None
        
        try:
            headers = {
                "Authorization": f"Token {self.replicate_token}",
                "Content-Type": "application/json"
            }
            
            # Utilisation du modèle Stable Diffusion XL
            payload = {
                "version": "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                "input": {
                    "prompt": prompt,
                    "negative_prompt": "blurry, low quality, distorted, ugly, deformed",
                    "width": 512,
                    "height": 512,
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5,
                    "scheduler": "K_EULER"
                }
            }
            
            # Créer la prédiction
            response = requests.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 201:
                prediction = response.json()
                prediction_url = prediction['urls']['get']
                
                # Polling pour attendre la génération
                for attempt in range(30):  # Max 5 minutes d'attente
                    time.sleep(10)
                    
                    status_response = requests.get(prediction_url, headers=headers, timeout=30)
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        
                        if status_data['status'] == 'succeeded':
                            if status_data.get('output'):
                                image_url = status_data['output'][0]
                                
                                # Télécharger l'image
                                img_response = requests.get(image_url, timeout=60)
                                if img_response.status_code == 200:
                                    image = Image.open(io.BytesIO(img_response.content))
                                    return image
                            break
                        elif status_data['status'] == 'failed':
                            logger.error(f"Replicate failed: {status_data.get('error', 'Unknown error')}")
                            break
                        elif status_data['status'] in ['starting', 'processing']:
                            continue
                        else:
                            break
                            
            return None
            
        except Exception as e:
            logger.error(f"Replicate exception: {e}")
            return None

    def _generate_local_diffusion(self, prompt: str) -> Optional[Image.Image]:
        """Génère une image avec Stable Diffusion local (Automatic1111)."""
        try:
            # Vérifier si l'instance locale est disponible
            test_response = requests.get("http://127.0.0.1:7860/", timeout=5)
            if test_response.status_code != 200:
                return None
            
            payload = {
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, distorted, ugly, deformed, text, watermark, signature",
                "steps": 25,
                "cfg_scale": 7.5,
                "width": 512,
                "height": 512,
                "sampler_name": "DPM++ 2M Karras",
                "batch_size": 1,
                "n_iter": 1,
                "seed": -1,
                "restore_faces": False,
                "tiling": False
            }
            
            response = requests.post(
                "http://127.0.0.1:7860/sdapi/v1/txt2img",
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('images'):
                    # Décoder l'image base64
                    image_data = base64.b64decode(result['images'][0])
                    image = Image.open(io.BytesIO(image_data))
                    return image
            
            return None
            
        except Exception as e:
            logger.error(f"Local Stable Diffusion exception: {e}")
            return None
        
    def _generate_runware(self, prompt: str) -> Optional[Image.Image]:
        """Génère une image avec Runware API (très fiable et rapide)."""
        runware_key = st.secrets.get("RUNWARE_API_KEY", "")
        if not runware_key:
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {runware_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "positivePrompt": prompt,
                "negativePrompt": "blurry, low quality, distorted, ugly, deformed, text, watermark, signature",
                "height": 512,
                "width": 512,
                "model": "runware:100@1",  # Modèle par défaut très rapide
                "steps": 20,
                "CFGScale": 7.0,
                "numberResults": 1
            }
            
            response = requests.post(
                "https://api.runware.ai/v1/images/generate",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("images") and len(result["images"]) > 0:
                    image_url = result["images"][0]["imageURL"]
                    
                    # Télécharger l'image
                    img_response = requests.get(image_url, timeout=30)
                    if img_response.status_code == 200:
                        image = Image.open(io.BytesIO(img_response.content))
                        logger.info("✅ Image générée avec Runware")
                        return image
            else:
                logger.error(f"Runware error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Runware exception: {e}")
            return None

    def _generate_stablediffusionapi(self, prompt: str) -> Optional[Image.Image]:
        """Génère une image avec StableDiffusionAPI.com (service gratuit)."""
        api_key = st.secrets.get("STABLEDIFFUSION_API_KEY", "")
        if not api_key:
            return None
        
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "key": api_key,
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, distorted, ugly, deformed, text, watermark",
                "width": "512",
                "height": "512",
                "samples": "1",
                "num_inference_steps": "25",
                "safety_checker": "yes",
                "enhance_prompt": "yes",
                "guidance_scale": 7.5,
                "webhook": None,
                "track_id": None
            }
            
            response = requests.post(
                "https://stablediffusionapi.com/api/v3/text2img",
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success" and result.get("output"):
                    image_url = result["output"][0]
                    
                    # Télécharger l'image
                    img_response = requests.get(image_url, timeout=30)
                    if img_response.status_code == 200:
                        image = Image.open(io.BytesIO(img_response.content))
                        logger.info("✅ Image générée avec StableDiffusionAPI")
                        return image
            else:
                logger.error(f"StableDiffusionAPI error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"StableDiffusionAPI exception: {e}")
            return None

    def _generate_pollinations(self, prompt: str) -> Optional[Image.Image]:
        """Génère une image avec Pollinations AI (100% gratuit, pas de clé requise)."""
        try:
            # Pollinations ne nécessite pas d'API key !
            import urllib.parse
            
            # Encoder le prompt pour l'URL
            encoded_prompt = urllib.parse.quote(prompt)
            
            # URL de l'API Pollinations (gratuite)
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&model=flux&seed={random.randint(1, 1000000)}"
            
            logger.debug(f"Pollinations URL: {image_url}")
            
            # Télécharger directement l'image
            response = requests.get(image_url, timeout=60)
            
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                logger.info("✅ Image générée avec Pollinations AI (gratuit)")
                return image
            else:
                logger.error(f"Pollinations error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Pollinations exception: {e}")
            return None

    def _generate_prodia(self, prompt: str) -> Optional[Image.Image]:
        """Génère une image avec Prodia API (gratuit)."""
        try:
            # Prodia API gratuite
            payload = {
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, distorted, ugly, deformed",
                "model": "sd_xl_base_1.0.safetensors [be9edd61]",
                "steps": 25,
                "cfg_scale": 7,
                "seed": -1,
                "upscale": False,
                "sampler": "DPM++ 2M Karras"
            }
            
            # Démarrer la génération
            response = requests.post(
                "https://api.prodia.com/v1/sd/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                job_id = result.get("job")
                
                if job_id:
                    # Attendre que l'image soit prête
                    for attempt in range(20):  # Max 2 minutes
                        time.sleep(6)
                        
                        status_response = requests.get(
                            f"https://api.prodia.com/v1/job/{job_id}",
                            timeout=10
                        )
                        
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            
                            if status_data.get("status") == "succeeded":
                                image_url = status_data.get("imageUrl")
                                if image_url:
                                    # Télécharger l'image finale
                                    img_response = requests.get(image_url, timeout=30)
                                    if img_response.status_code == 200:
                                        image = Image.open(io.BytesIO(img_response.content))
                                        logger.info("✅ Image générée avec Prodia")
                                        return image
                                break
                            elif status_data.get("status") == "failed":
                                logger.error("Prodia generation failed")
                                break
            
            return None
            
        except Exception as e:
            logger.error(f"Prodia exception: {e}")
            return None

    def create_enhanced_placeholder_image(self, dream_text: str) -> Image.Image:
        """
        Crée une image placeholder améliorée avec plus d'éléments visuels.
        """
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
        import random
        import math
        
        # Configuration
        width, height = 512, 512
        image = Image.new('RGB', (width, height), color=(15, 15, 30))
        draw = ImageDraw.Draw(image)
        
        # Analyse des mots-clés pour adapter les couleurs
        keywords_colors = {
            'princess': [(255, 192, 203), (147, 112, 219), (255, 215, 0)],
            'castle': [(105, 105, 105), (128, 128, 128), (169, 169, 169)],
            'forest': [(34, 139, 34), (0, 100, 0), (46, 125, 50)],
            'ocean': [(0, 119, 190), (0, 150, 255), (64, 224, 208)],
            'fire': [(255, 69, 0), (255, 140, 0), (255, 215, 0)],
            'night': [(25, 25, 112), (72, 61, 139), (123, 104, 238)],
            'magic': [(138, 43, 226), (186, 85, 211), (147, 112, 219)]
        }
        
        # Détection des mots-clés
        text_lower = dream_text.lower()
        selected_colors = [(138, 43, 226), (75, 0, 130), (255, 215, 0)]
        
        for keyword, colors in keywords_colors.items():
            if keyword in text_lower:
                selected_colors = colors
                break
        
        # Création d'un fond dégradé radial simple
        center_x, center_y = width // 2, height // 2
        for y in range(height):
            for x in range(width):
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_distance = math.sqrt(center_x**2 + center_y**2)
                factor = min(distance / max_distance, 1.0)
                
                # Interpolation simple entre 2 couleurs
                color1 = selected_colors[0]
                color2 = selected_colors[1] if len(selected_colors) > 1 else selected_colors[0]
                
                r = int(color1[0] * (1 - factor) + color2[0] * factor)
                g = int(color1[1] * (1 - factor) + color2[1] * factor)
                b = int(color1[2] * (1 - factor) + color2[2] * factor)
                
                try:
                    image.putpixel((x, y), (r, g, b))
                except:
                    continue
        
        # Ajout de formes simples
        for i in range(5):
            x = random.randint(50, width - 50)
            y = random.randint(50, height - 50)
            size = random.randint(20, 60)
            color = random.choice(selected_colors)
            
            draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], 
                        fill=(*color, 100))
        
        # Ajout du texte
        try:
            font_large = ImageFont.truetype("arial.ttf", 28)
            font_small = ImageFont.truetype("arial.ttf", 16)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Titre
        title = "🌙 Rêve Synthétisé"
        title_bbox = draw.textbbox((0, 0), title, font=font_large)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        title_y = 30
        
        draw.rectangle([title_x-10, title_y-5, title_x+title_width+10, title_y+35], 
                    fill=(0, 0, 0, 150))
        draw.text((title_x, title_y), title, font=font_large, fill=(255, 255, 255))
        
        # Texte du rêve
        preview_text = dream_text[:60] + "..." if len(dream_text) > 60 else dream_text
        preview_bbox = draw.textbbox((0, 0), preview_text, font=font_small)
        preview_width = preview_bbox[2] - preview_bbox[0]
        preview_x = (width - preview_width) // 2
        preview_y = height - 60
        
        draw.rectangle([preview_x-10, preview_y-5, preview_x+preview_width+10, preview_y+25], 
                    fill=(0, 0, 0, 150))
        draw.text((preview_x, preview_y), preview_text, font=font_small, fill=(255, 255, 255))
        
        logger.info("✅ Image placeholder améliorée créée")
        return image

    def save_dream(self, dream_data: Dict[str, Any]) -> bool:
        """
        Sauvegarde un rêve dans le stockage local (fichier JSON).
        Inclut maintenant les prompts générés.
        """
        logger.debug(f"💾 Sauvegarde du rêve: {dream_data.get('title', 'Sans titre')}")
        
        try:
            dreams_file = Path("dreams_data.json")
            
            # Chargement des rêves existants
            if dreams_file.exists():
                with open(dreams_file, "r", encoding="utf-8") as f:
                    dreams = json.load(f)
                    logger.debug(f"📚 Chargement de {len(dreams)} rêves existants")
            else:
                dreams = []
                logger.debug("📝 Création d'un nouveau fichier de rêves")
            
            # Ajout de métadonnées supplémentaires
            dream_data.update({
                "id": dream_data.get("id", f"dream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                "created_at": datetime.now().isoformat(),
                "version": "1.0.0"
            })
            
            # Ajout du nouveau rêve
            dreams.append(dream_data)
            
            # Sauvegarde avec indentation pour la lisibilité
            with open(dreams_file, "w", encoding="utf-8") as f:
                json.dump(dreams, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Rêve sauvegardé avec succès (total: {len(dreams)} rêves)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde: {e}")
            return False

    def load_dreams(self) -> List[Dict[str, Any]]:
        """
        Charge tous les rêves sauvegardés depuis le fichier JSON.
        """
        logger.debug("📖 Chargement des rêves sauvegardés")
        
        try:
            dreams_file = Path("dreams_data.json")
            
            if dreams_file.exists():
                with open(dreams_file, "r", encoding="utf-8") as f:
                    dreams = json.load(f)
                    logger.info(f"✅ Chargement de {len(dreams)} rêves réussi")
                    return dreams
            else:
                logger.debug("📝 Aucun fichier de rêves trouvé")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ Erreur de format JSON lors du chargement: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des rêves: {e}")
            return []

    def save_dream_image(self, image: Image.Image, dream_id: str) -> Optional[str]:
        """
        Sauvegarde une image de rêve sur le disque.
        """
        logger.debug(f"🖼️ Sauvegarde de l'image pour le rêve {dream_id}")
        
        try:
            # Création du dossier images
            images_dir = Path("dream_images")
            images_dir.mkdir(exist_ok=True)
            
            # Nom du fichier avec timestamp pour éviter les conflits
            image_path = images_dir / f"dream_{dream_id}.png"
            
            # Sauvegarde de l'image en PNG pour préserver la qualité
            image.save(image_path, "PNG", optimize=True)
            
            logger.info(f"✅ Image sauvegardée: {image_path} ({image.size})")
            return str(image_path)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde de l'image: {e}")
            return None
        
    def _generate_pollinations_video(self, prompt: str, image_path: Optional[str] = None) -> Optional[str]:
        """Génère une vidéo avec Pollinations (gratuit, comme les images)."""
        try:
            import urllib.parse
            
            # Nettoyer le prompt pour la vidéo
            clean_prompt = prompt.replace('"', '').replace('\n', ' ')
            clean_prompt = ' '.join(clean_prompt.split())[:150]  # Limiter
            
            # Encoder le prompt pour l'URL
            encoded_prompt = urllib.parse.quote(clean_prompt)
            
            # URL de l'API Pollinations pour vidéo (basée sur leur structure)
            seed = random.randint(1, 1000000)
            
            # Essayer différentes URLs possibles pour la vidéo
            video_urls = [
                f"https://video.pollinations.ai/prompt/{encoded_prompt}?seed={seed}",
                f"https://image.pollinations.ai/prompt/{encoded_prompt}?seed={seed}&format=gif&animation=true",
                f"https://pollinations.ai/api/video?prompt={encoded_prompt}&seed={seed}",
            ]
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            for video_url in video_urls:
                try:
                    logger.debug(f"Tentative URL vidéo: {video_url}")
                    
                    response = requests.get(video_url, headers=headers, timeout=90)
                    
                    if response.status_code == 200:
                        # Vérifier le type de contenu
                        content_type = response.headers.get('content-type', '')
                        
                        if any(video_type in content_type for video_type in ['video/', 'image/gif']):
                            # Sauvegarder la vidéo
                            videos_dir = Path("dream_videos")
                            videos_dir.mkdir(exist_ok=True)
                            
                            file_ext = '.mp4' if 'video/' in content_type else '.gif'
                            video_path = videos_dir / f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_ext}"
                            
                            with open(video_path, "wb") as f:
                                f.write(response.content)
                            
                            logger.info(f"✅ Vidéo générée avec Pollinations: {video_path}")
                            return str(video_path)
                        
                except Exception as e:
                    logger.debug(f"Échec URL {video_url}: {e}")
                    continue
            
            # Si aucune URL vidéo ne fonctionne, créer une vidéo à partir de l'image
            if image_path:
                return self._create_video_from_image_pollinations(image_path, prompt)
            
            return None
            
        except Exception as e:
            logger.error(f"Pollinations Video exception: {e}")
            return None

    def _create_video_from_image_pollinations(self, image_path: str, prompt: str) -> Optional[str]:
        """Crée une vidéo simple à partir d'une image avec effet de parallaxe."""
        try:
            from PIL import Image, ImageFilter, ImageEnhance
            import math
            
            logger.debug("Création d'une vidéo avec effet de parallaxe à partir de l'image")
            
            # Charger l'image
            image = Image.open(image_path)
            
            # Redimensionner pour la vidéo
            video_width, video_height = 512, 512
            image = image.resize((video_width + 50, video_height + 50), Image.Resampling.LANCZOS)
            
            # Créer plusieurs frames avec effet de mouvement
            frames = []
            num_frames = 30  # 30 frames pour ~1 seconde à 30fps
            
            for i in range(num_frames):
                # Effet de parallaxe simple (mouvement lent)
                progress = i / num_frames
                
                # Calcul du décalage
                offset_x = int(math.sin(progress * math.pi * 2) * 10)
                offset_y = int(math.cos(progress * math.pi * 2) * 5)
                
                # Créer le frame
                frame = Image.new('RGB', (video_width, video_height), (0, 0, 0))
                
                # Coller l'image avec offset
                paste_x = offset_x + 25
                paste_y = offset_y + 25
                
                frame.paste(image, (paste_x, paste_y))
                
                # Ajouter un léger effet de luminosité qui varie
                brightness_factor = 0.9 + 0.2 * math.sin(progress * math.pi * 4)
                enhancer = ImageEnhance.Brightness(frame)
                frame = enhancer.enhance(brightness_factor)
                
                frames.append(frame)
            
            # Sauvegarder comme GIF animé
            videos_dir = Path("dream_videos")
            videos_dir.mkdir(exist_ok=True)
            
            gif_path = videos_dir / f"animated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
            
            # Créer le GIF
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=100,  # 100ms par frame = 10fps
                loop=0,  # Boucle infinie
                optimize=True
            )
            
            logger.info(f"✅ Vidéo GIF créée: {gif_path}")
            return str(gif_path)
            
        except Exception as e:
            logger.error(f"Erreur création vidéo parallaxe: {e}")
            return None

    def _generate_pollinations_music(self, prompt: str) -> Optional[str]:
        """Génère de la musique avec Pollinations (si disponible)."""
        try:
            import urllib.parse
            
            # Nettoyer le prompt musical
            clean_prompt = prompt.replace('"', '').replace('\n', ' ')
            clean_prompt = ' '.join(clean_prompt.split())[:100]
            
            # Encoder le prompt
            encoded_prompt = urllib.parse.quote(clean_prompt)
            
            # URLs possibles pour l'audio Pollinations
            audio_urls = [
                f"https://audio.pollinations.ai/prompt/{encoded_prompt}",
                f"https://text.pollinations.ai/{encoded_prompt}?model=audio&voice=ambient",
                f"https://pollinations.ai/api/audio?prompt={encoded_prompt}&duration=30",
            ]
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            for audio_url in audio_urls:
                try:
                    logger.debug(f"Tentative URL audio: {audio_url}")
                    
                    response = requests.get(audio_url, headers=headers, timeout=120)
                    
                    if response.status_code == 200:
                        content_type = response.headers.get('content-type', '')
                        
                        if 'audio/' in content_type:
                            # Sauvegarder l'audio
                            music_dir = Path("dream_music")
                            music_dir.mkdir(exist_ok=True)
                            
                            file_ext = '.mp3' if 'mp3' in content_type else '.wav'
                            music_path = music_dir / f"music_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_ext}"
                            
                            with open(music_path, "wb") as f:
                                f.write(response.content)
                            
                            logger.info(f"✅ Musique générée avec Pollinations: {music_path}")
                            return str(music_path)
                    
                except Exception as e:
                    logger.debug(f"Échec URL audio {audio_url}: {e}")
                    continue
            
            # Fallback : créer un fichier audio simple avec des tonalités
            return self._create_simple_ambient_audio(prompt)
            
        except Exception as e:
            logger.error(f"Pollinations Music exception: {e}")
            return None

    def _create_simple_ambient_audio(self, prompt: str) -> Optional[str]:
        """Crée un audio ambiant simple basé sur le prompt."""
        try:
            import numpy as np
            import wave
            
            logger.debug("Création d'un audio ambiant simple")
            
            # Analyser le prompt pour déterminer les fréquences
            text_lower = prompt.lower()
            
            # Fréquences de base selon l'ambiance
            if any(word in text_lower for word in ['peaceful', 'calm', 'serene', 'paisible']):
                base_freqs = [220, 330, 440]  # Notes douces
            elif any(word in text_lower for word in ['mysterious', 'dark', 'mystérieux']):
                base_freqs = [110, 165, 220]  # Notes graves
            elif any(word in text_lower for word in ['magical', 'fairy', 'magique']):
                base_freqs = [440, 550, 660]  # Notes éthérées
            else:
                base_freqs = [262, 330, 392]  # Do, Mi, Sol (accord majeur)
            
            # Paramètres audio
            sample_rate = 44100
            duration = 10  # 10 secondes
            samples = int(sample_rate * duration)
            
            # Générer le signal audio
            t = np.linspace(0, duration, samples, False)
            audio = np.zeros(samples)
            
            # Ajouter les harmoniques
            for i, freq in enumerate(base_freqs):
                # Onde sinusoïdale avec enveloppe
                amplitude = 0.3 / len(base_freqs) * (1 - i * 0.2)
                
                # Enveloppe ADSR simplifiée
                envelope = np.ones_like(t)
                fade_samples = int(sample_rate * 2)  # 2 secondes de fade
                
                # Fade in
                envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
                # Fade out
                envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
                
                # Générer la tonalité
                wave_signal = amplitude * np.sin(2 * np.pi * freq * t) * envelope
                
                # Ajouter un peu de modulation pour l'effet ambiant
                modulation = 0.1 * np.sin(2 * np.pi * 0.5 * t)  # Modulation lente
                wave_signal *= (1 + modulation)
                
                audio += wave_signal
            
            # Normaliser l'audio
            audio = audio / np.max(np.abs(audio)) * 0.8
            
            # Convertir en 16-bit
            audio_16bit = (audio * 32767).astype(np.int16)
            
            # Sauvegarder le fichier WAV
            music_dir = Path("dream_music")
            music_dir.mkdir(exist_ok=True)
            
            wav_path = music_dir / f"ambient_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            
            with wave.open(str(wav_path), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_16bit.tobytes())
            
            logger.info(f"✅ Audio ambiant créé: {wav_path}")
            return str(wav_path)
            
        except Exception as e:
            logger.error(f"Erreur création audio ambiant: {e}")
            return None

    # MÉTHODES PRINCIPALES SIMPLIFIÉES

    def generate_video(self, dream_text: str, image_path: Optional[str] = None) -> Optional[str]:
        """
        Génère une vidéo à partir du texte du rêve - VERSION SIMPLIFIÉE POLLINATIONS.
        """
        logger.debug(f"🎬 Génération de vidéo pour: {dream_text[:50]}...")
        
        # Génération du prompt vidéo optimisé
        video_prompt = self.generate_video_prompt(dream_text)
        logger.info(f"Prompt vidéo utilisé: {video_prompt}")
        
        # Essayer Pollinations en premier
        try:
            logger.debug("📡 Tentative vidéo Pollinations")
            video_path = self._generate_pollinations_video(video_prompt, image_path)
            
            if video_path:
                logger.info("✅ Vidéo générée avec succès via Pollinations")
                return video_path
            else:
                logger.warning("⚠️ Pollinations vidéo n'a pas fonctionné")
                
        except Exception as e:
            logger.error(f"❌ Erreur vidéo Pollinations: {e}")
        
        # Fallback : créer une vidéo à partir de l'image
        if image_path and Path(image_path).exists():
            logger.debug("🔄 Fallback: création vidéo à partir de l'image")
            return self._create_video_from_image_pollinations(image_path, video_prompt)
        
        logger.warning("❌ Impossible de générer une vidéo")
        return None

    def generate_music(self, dream_text: str) -> Optional[str]:
        """
        Génère de la musique à partir du texte du rêve - VERSION SIMPLIFIÉE POLLINATIONS.
        """
        logger.debug(f"🎵 Génération de musique pour: {dream_text[:50]}...")
        
        # Génération du prompt musical optimisé
        music_prompt = self.generate_music_prompt(dream_text)
        logger.info(f"Prompt musical utilisé: {music_prompt}")
        
        # Essayer Pollinations en premier
        try:
            logger.debug("📡 Tentative musique Pollinations")
            music_path = self._generate_pollinations_music(music_prompt)
            
            if music_path:
                logger.info("✅ Musique générée avec succès via Pollinations")
                return music_path
            else:
                logger.warning("⚠️ Pollinations musique n'a pas fonctionné")
                
        except Exception as e:
            logger.error(f"❌ Erreur musique Pollinations: {e}")
        
        # Fallback : créer un audio ambiant simple
        logger.debug("🔄 Fallback: création audio ambiant simple")
        return self._create_simple_ambient_audio(music_prompt)

    # TEST RAPIDE POUR VÉRIFIER LES FONCTIONNALITÉS

    def test_video_music_generation(self):
        """Test rapide des fonctionnalités vidéo et musique."""
        try:
            logger.info("🧪 Test des fonctionnalités multimédia...")
            
            test_prompt = "beautiful princess in magical forest"
            
            # Test vidéo
            logger.info("🎬 Test génération vidéo...")
            video_result = self.generate_video(test_prompt)
            
            if video_result:
                logger.info(f"✅ Test vidéo réussi: {video_result}")
            else:
                logger.warning("⚠️ Test vidéo échoué")
            
            # Test musique
            logger.info("🎵 Test génération musique...")
            music_result = self.generate_music(test_prompt)
            
            if music_result:
                logger.info(f"✅ Test musique réussi: {music_result}")
            else:
                logger.warning("⚠️ Test musique échoué")
            
            return video_result is not None or music_result is not None
            
        except Exception as e:
            logger.error(f"❌ Erreur test multimédia: {e}")
            return False

# ================================
# FONCTIONS UTILITAIRES UI
# ================================

def create_emotion_chart(emotions: Dict[str, float]) -> None:
    """
    Crée un graphique des émotions détectées avec couleurs personnalisées.
    
    Args:
        emotions (Dict[str, float]): Dictionnaire des émotions et leurs scores
    """
    logger.debug("📊 Création du graphique des émotions")
    
    # Couleurs personnalisées pour chaque émotion
    emotion_colors = {
        "heureux": "#FFD700",      # Or
        "stressant": "#FF6B6B",    # Rouge
        "neutre": "#95A5A6",       # Gris
        "triste": "#3498DB",       # Bleu
        "excitant": "#E74C3C",     # Rouge vif
        "paisible": "#2ECC71",     # Vert
        "angoissant": "#8E44AD",   # Violet
        "mystérieux": "#34495E"    # Bleu foncé
    }
    
    # Filtrage des émotions avec un score significatif (> 0.1)
    significant_emotions = {k: v for k, v in emotions.items() if v > 0.1}
    
    if significant_emotions:
        logger.debug(f"📈 Affichage de {len(significant_emotions)} émotions significatives")
        
        # Création du DataFrame pour le graphique
        df = pd.DataFrame(list(significant_emotions.items()), columns=["Émotion", "Intensité"])
        df = df.sort_values("Intensité", ascending=True)  # Tri croissant pour un meilleur affichage
        
        # Affichage du graphique en barres horizontales
        st.bar_chart(
            data=df.set_index("Émotion")["Intensité"],
            color="#FFD700",  # Couleur dorée pour l'uniformité
            height=300
        )
        
        # Affichage des scores détaillés avec couleurs
        st.write("**Détail des émotions détectées:**")
        
        # Tri par intensité décroissante pour l'affichage
        sorted_emotions = sorted(significant_emotions.items(), key=lambda x: x[1], reverse=True)
        
        for emotion, score in sorted_emotions:
            color = emotion_colors.get(emotion, "#95A5A6")
            percentage = score * 100
            
            # Affichage avec barre de progression colorée
            st.markdown(f"""
            <div style="margin: 10px 0;">
                <div style="display: flex; align-items: center;">
                    <strong style="width: 120px;">{emotion.capitalize()}:</strong>
                    <div style="
                        flex: 1; 
                        background: linear-gradient(90deg, {color} {percentage}%, #2C3E50 {percentage}%);
                        height: 20px;
                        border-radius: 10px;
                        margin: 0 10px;
                        position: relative;
                    ">
                        <span style="
                            position: absolute;
                            right: 10px;
                            color: white;
                            font-weight: bold;
                            line-height: 20px;
                            text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
                        ">{percentage:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        logger.debug("✅ Graphique des émotions affiché")
    else:
        st.info("🤖 Aucune émotion significative détectée dans ce rêve.")
        logger.debug("ℹ️ Aucune émotion significative à afficher")


def filter_dreams(dreams: List[Dict], emotion_filter: str, date_filter: str, search_text: str) -> List[Dict]:
    """
    Filtre la liste des rêves selon les critères spécifiés.
    
    Args:
        dreams (List[Dict]): Liste des rêves à filtrer
        emotion_filter (str): Filtre par émotion dominante
        date_filter (str): Filtre par période temporelle
        search_text (str): Texte de recherche dans le contenu
        
    Returns:
        List[Dict]: Rêves filtrés selon les critères
    """
    logger.debug(f"🔍 Filtrage des rêves: émotion={emotion_filter}, date={date_filter}, recherche='{search_text}'")
    
    filtered = dreams.copy()
    initial_count = len(filtered)
    
    # Filtre par émotion dominante
    if emotion_filter != "Toutes":
        filtered = []
        for dream in dreams:
            if dream.get("emotions"):
                # Trouve l'émotion dominante (score le plus élevé)
                dominant_emotion = max(dream["emotions"], key=dream["emotions"].get)
                if dominant_emotion == emotion_filter.lower():
                    filtered.append(dream)
        
        logger.debug(f"🎭 Après filtre émotion '{emotion_filter}': {len(filtered)} rêves")
    
    # Filtre par texte de recherche
    if search_text and search_text.strip():
        search_lower = search_text.lower().strip()
        text_filtered = []
        
        for dream in filtered:
            # Recherche dans le titre, le texte et les tags
            title_match = search_lower in dream.get("title", "").lower()
            text_match = search_lower in dream.get("text", "").lower()
            tags_match = any(search_lower in tag.lower() for tag in dream.get("tags", []))
            
            if title_match or text_match or tags_match:
                text_filtered.append(dream)
        
        filtered = text_filtered
        logger.debug(f"🔤 Après filtre texte '{search_text}': {len(filtered)} rêves")
    
    # Filtre par date (implémentation basique)
    if date_filter != "Toutes":
        date_filtered = []
        now = datetime.now()
        
        for dream in filtered:
            try:
                dream_date = datetime.fromisoformat(dream.get("date", ""))
                
                if date_filter == "Dernière semaine":
                    if (now - dream_date).days <= 7:
                        date_filtered.append(dream)
                elif date_filter == "Dernier mois":
                    if (now - dream_date).days <= 30:
                        date_filtered.append(dream)
                elif date_filter == "Dernière année":
                    if (now - dream_date).days <= 365:
                        date_filtered.append(dream)
                        
            except (ValueError, TypeError):
                logger.warning(f"⚠️ Date invalide pour le rêve {dream.get('id', 'unknown')}")
                continue
        
        filtered = date_filtered
        logger.debug(f"📅 Après filtre date '{date_filter}': {len(filtered)} rêves")
    
    logger.info(f"✅ Filtrage terminé: {len(filtered)}/{initial_count} rêves conservés")
    return filtered


def display_dream_card(dream: Dict[str, Any]) -> None:
    """
    Affiche une carte détaillée pour un rêve avec toutes ses informations.
    
    Args:
        dream (Dict[str, Any]): Données du rêve à afficher
    """
    dream_id = dream.get('id', 'unknown')
    dream_title = dream.get('title', 'Sans titre')
    dream_date = dream.get('date', 'Date inconnue')
    
    logger.debug(f"🃏 Affichage de la carte du rêve: {dream_id}")
    
    # Format de la date pour l'affichage
    try:
        formatted_date = datetime.fromisoformat(dream_date).strftime("%d/%m/%Y à %H:%M")
    except (ValueError, TypeError):
        formatted_date = dream_date[:10] if len(dream_date) > 10 else dream_date
    
    # Icône basée sur l'émotion dominante
    emotion_icons = {
        "heureux": "😊",
        "stressant": "😰",
        "neutre": "😐",
        "triste": "😢",
        "excitant": "🤩",
        "paisible": "😌",
        "angoissant": "😱",
        "mystérieux": "🤔"
    }
    
    # Détermination de l'émotion dominante
    dominant_emotion = "neutre"
    if dream.get("emotions"):
        dominant_emotion = max(dream["emotions"], key=dream["emotions"].get)
    
    icon = emotion_icons.get(dominant_emotion, "🌙")
    
    # Affichage de la carte dans un expander
    with st.expander(f"{icon} {dream_title} - {formatted_date}", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Texte du rêve
            st.markdown("**📖 Récit du rêve:**")
            dream_text = dream.get("text", "Aucun texte disponible")
            
            # Limitation de l'affichage pour les longs textes
            if len(dream_text) > 500:
                st.write(dream_text[:500] + "...")
                with st.expander("Voir le texte complet"):
                    st.write(dream_text)
            else:
                st.write(dream_text)
            
            # Affichage du prompt s'il existe
            if dream.get("image_prompt"):
                with st.expander("🎯 Prompt utilisé pour l'image", expanded=False):
                    st.code(dream["image_prompt"], language="text")
                    st.caption("Prompt qui a généré l'image de ce rêve")
            
            # Tags si disponibles
            if dream.get("tags"):
                st.markdown("**🏷️ Tags:**")
                tags_html = " ".join([f'<span style="background-color: #FFD700; color: black; padding: 2px 8px; border-radius: 12px; margin: 2px; font-size: 0.8em;">{tag}</span>' for tag in dream["tags"]])
                st.markdown(tags_html, unsafe_allow_html=True)
        
        with col2:
            # Image si disponible
            image_path = dream.get("image_path")
            if image_path and Path(image_path).exists():
                try:
                    image = Image.open(image_path)
                    st.image(image, caption="Image du rêve", use_container_width=True)
                    
                    # Bouton de téléchargement de l'image
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format="PNG")
                    st.download_button(
                        label="📥 Télécharger l'image",
                        data=img_buffer.getvalue(),
                        file_name=f"reve_{dream_id}.png",
                        mime="image/png",
                        key=f"download_img_{dream_id}"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Impossible de charger l'image: {e}")
                    logger.error(f"❌ Erreur chargement image {image_path}: {e}")
            else:
                st.info("🖼️ Aucune image disponible")
            
            # Émotions détectées
            if dream.get("emotions"):
                st.markdown("**🎭 Émotions détectées:**")
                emotions = dream["emotions"]
                
                # Tri par score décroissant et affichage des 3 principales
                sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                
                for emotion, score in sorted_emotions:
                    if score > 0.1:  # Seulement les émotions significatives
                        icon = emotion_icons.get(emotion, "🔹")
                        st.write(f"{icon} **{emotion.capitalize()}**: {score*100:.0f}%")
        
        # Métadonnées du rêve
        st.markdown("---")
        col_meta1, col_meta2, col_meta3 = st.columns(3)
        
        with col_meta1:
            st.metric("📅 Date", formatted_date)
        
        with col_meta2:
            word_count = len(dream.get("text", "").split())
            st.metric("📝 Mots", word_count)
        
        with col_meta3:
            emotion_count = len([e for e in dream.get("emotions", {}).values() if e > 0.1])
            st.metric("🎭 Émotions", emotion_count)


# ================================
# FONCTIONS DE GESTION DES PAGES
# ================================

def handle_new_dream(synthesizer) -> None:
    """
    Gère la page de création d'un nouveau rêve.
    
    Args:
        synthesizer: Instance du synthétiseur de rêves
    """
    logger.debug("🆕 Affichage de la page nouveau rêve")
    
    st.header("🎤 Racontez votre rêve")
    st.markdown("Transformez votre rêve en art visuel avec l'intelligence artificielle !")
    
    # Choix du mode d'entrée
    input_mode = st.radio(
        "💭 Comment souhaitez-vous raconter votre rêve ?",
        ["📝 Saisie texte", "📁 Upload fichier audio"],
        help="Choisissez la méthode qui vous convient le mieux"
    )
    
    dream_text = ""
    
    if input_mode == "📝 Saisie texte":
        # Saisie directe de texte
        st.markdown("### ✍️ Tapez votre rêve")
        dream_text = st.text_area(
            "Décrivez votre rêve en détail:",
            height=200,
            placeholder="Il était une fois, dans mon rêve... Je me trouvais dans un endroit magique où...",
            help="Plus votre description est détaillée, plus l'image générée sera précise !"
        )
        
        if dream_text:
            word_count = len(dream_text.split())
            char_count = len(dream_text)
            st.caption(f"📊 {word_count} mots • {char_count} caractères")
            
        logger.debug(f"📝 Texte saisi: {len(dream_text)} caractères")
    
    elif input_mode == "📁 Upload fichier audio":
        # Upload de fichier audio
        st.markdown("### 📂 Upload d'un fichier audio")
        
        audio_file = st.file_uploader(
            "Choisissez un fichier audio:",
            type=["wav", "mp3", "m4a", "ogg"],
            help="Formats supportés: WAV, MP3, M4A, OGG (max 200MB)"
        )
        
        if audio_file is not None:
            logger.info(f"📁 Fichier audio uploadé: {audio_file.name} ({audio_file.size} bytes)")
            
            # Affichage des informations du fichier
            file_size_mb = audio_file.size / (1024 * 1024)
            st.success(f"✅ Fichier chargé: **{audio_file.name}** ({file_size_mb:.1f} MB)")
            
            # Lecteur audio pour prévisualisation
            st.audio(audio_file, format=f"audio/{audio_file.type.split('/')[-1]}")
            
            # Bouton de transcription
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if st.button("🔄 Transcrire l'audio", type="primary", use_container_width=True):
                    with st.spinner("🎯 Transcription en cours..."):
                        logger.info("🔄 Début de la transcription audio")
                        
                        # Reset du pointeur de fichier
                        audio_file.seek(0)
                        
                        dream_text = synthesizer.transcribe_audio(audio_file)
                        
                        if dream_text and not any(error in dream_text.lower() for error in ["erreur", "impossible"]):
                            st.success("✅ Transcription réussie!")
                            logger.info("✅ Transcription audio réussie")
                        else:
                            st.error(f"❌ Erreur de transcription: {dream_text}")
                            logger.error(f"❌ Échec transcription: {dream_text}")
            
            with col2:
                if dream_text:
                    st.info("💡 Vous pouvez modifier le texte ci-dessous avant de continuer")
            
            # Zone d'édition du texte transcrit
            if dream_text and not any(error in dream_text.lower() for error in ["erreur", "impossible"]):
                st.markdown("### 📝 Texte transcrit")
                dream_text = st.text_area(
                    "Vérifiez et modifiez si nécessaire:",
                    value=dream_text,
                    height=150,
                    help="Vous pouvez corriger ou compléter la transcription"
                )
    
    # Traitement du rêve si le texte est disponible et suffisant
    if dream_text and len(dream_text.strip()) > 20:
        st.markdown("---")
        
        # Validation du contenu
        word_count = len(dream_text.split())
        if word_count < 5:
            st.warning("⚠️ Votre rêve semble un peu court. Ajoutez plus de détails pour une meilleure analyse.")
        else:
            # Bouton principal de traitement
            st.markdown("### ✨ Prêt à transformer votre rêve ?")
            
            if st.button("🎨 **Synthétiser le rêve**", type="primary", use_container_width=True):
                process_dream(synthesizer, dream_text)
    
    elif dream_text and len(dream_text.strip()) <= 20:
        st.warning("⚠️ Veuillez saisir au moins 20 caractères pour décrire votre rêve.")


def process_dream(synthesizer, dream_text: str) -> None:
    """
    Traite un rêve complet : analyse + image + vidéo + musique.
    """
    logger.info(f"🔄 Début du traitement complet du rêve: {dream_text[:50]}...")
    
    # Variables
    emotions = {}
    image = None
    video_path = None
    music_path = None
    generated_prompt = None
    
    # Barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Options de génération
    st.markdown("### 🎛️ Options de génération")
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    
    with col_opt1:
        generate_image = st.checkbox("🖼️ Générer une image", value=True)
    with col_opt2:
        generate_video = st.checkbox("🎬 Générer une vidéo", value=True)
    with col_opt3:
        generate_music = st.checkbox("🎵 Générer de la musique", value=True)
    
    # Colonnes d'affichage
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎭 Analyse émotionnelle")
        
        # Étape 1: Émotions
        with st.spinner("🧠 Analyse des émotions..."):
            status_text.text("🔍 Analyse émotionnelle...")
            progress_bar.progress(20)
            
            emotions = synthesizer.analyze_emotion(dream_text)
            
            if emotions:
                st.success("✅ Analyse émotionnelle terminée!")
                create_emotion_chart(emotions)
            else:
                st.error("❌ Erreur lors de l'analyse émotionnelle")
    
    with col2:
        st.subheader("🎨 Génération d'image")
        
        # Étape 2: Image
        if generate_image:
            with st.spinner("🖼️ Création de l'image..."):
                status_text.text("🎨 Génération de l'image...")
                progress_bar.progress(40)
                
                # Générer d'abord le prompt et l'afficher
                generated_prompt = synthesizer.generate_image_prompt(dream_text)
                
                # Affichage du prompt généré
                with st.expander("🎯 Prompt généré pour l'image", expanded=False):
                    st.markdown("**Prompt utilisé par l'IA :**")
                    st.code(generated_prompt, language="text")
                    st.caption("Ce prompt a été optimisé pour créer une image artistique de votre rêve")
                
                # Générer l'image avec l'ancienne méthode pour l'instant
                image = synthesizer.generate_image(dream_text)
                
                # Si generate_image retourne un tuple, on le gère
                if isinstance(image, tuple):
                    image, generated_prompt = image
                
                if image:
                    st.success("✅ Image générée!")
                    st.image(image, caption="🌙 Votre rêve visualisé", use_container_width=True)
                    
                    # Bouton téléchargement
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format="PNG")
                    st.download_button(
                        label="📥 Télécharger l'image",
                        data=img_buffer.getvalue(),
                        file_name=f"reve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                else:
                    st.error("❌ Erreur génération image")
        else:
            progress_bar.progress(40)
    
    # Section vidéo et musique
    st.markdown("---")
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.subheader("🎬 Génération de vidéo")
        
        # Étape 3: Vidéo
        if generate_video and image:
            with st.spinner("🎬 Création de la vidéo..."):
                status_text.text("🎬 Génération de la vidéo...")
                progress_bar.progress(70)
                
                # Générer et afficher le prompt vidéo
                video_prompt = synthesizer.generate_video_prompt(dream_text)
                
                with st.expander("🎬 Prompt généré pour la vidéo", expanded=False):
                    st.markdown("**Prompt vidéo utilisé :**")
                    st.code(video_prompt, language="text")
                    st.caption("Ce prompt guide la création de votre vidéo onirique")
                
                # Sauvegarder temporairement l'image
                temp_dir = Path("temp_images")
                temp_dir.mkdir(exist_ok=True)
                temp_image_path = temp_dir / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                image.save(temp_image_path)
                
                video_path = synthesizer.generate_video(dream_text, str(temp_image_path))
                
                if video_path and Path(video_path).exists():
                    st.success("✅ Vidéo générée!")
                    
                    # Afficher le GIF
                    if video_path.endswith('.gif'):
                        with open(video_path, "rb") as gif_file:
                            st.image(gif_file.read(), caption="🎬 Votre rêve animé")
                    
                    # Bouton téléchargement
                    with open(video_path, "rb") as f:
                        file_ext = "gif" if video_path.endswith('.gif') else "mp4"
                        mime_type = "image/gif" if video_path.endswith('.gif') else "video/mp4"
                        
                        st.download_button(
                            label="📥 Télécharger la vidéo",
                            data=f.read(),
                            file_name=f"reve_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                            mime=mime_type,
                            use_container_width=True
                        )
                else:
                    st.warning("⚠️ Impossible de créer la vidéo")
        elif generate_video and not image:
            st.info("ℹ️ Image requise pour générer la vidéo")
        else:
            progress_bar.progress(70)
    
    with col4:
        st.subheader("🎵 Génération de musique")
        
        # Étape 4: Musique
        if generate_music:
            with st.spinner("🎵 Création de la musique..."):
                status_text.text("🎵 Génération de la musique...")
                progress_bar.progress(90)
                
                # Générer et afficher le prompt musical
                music_prompt = synthesizer.generate_music_prompt(dream_text)
                
                with st.expander("🎵 Prompt généré pour la musique", expanded=False):
                    st.markdown("**Prompt musical utilisé :**")
                    st.code(music_prompt, language="text")
                    st.caption("Ce prompt guide la création de votre ambiance sonore")
                
                music_path = synthesizer.generate_music(dream_text)
                
                if music_path and Path(music_path).exists():
                    st.success("✅ Musique générée!")
                    
                    # Lecteur audio
                    with open(music_path, "rb") as audio_file:
                        st.audio(audio_file.read(), format="audio/wav")
                    
                    # Bouton téléchargement
                    with open(music_path, "rb") as f:
                        st.download_button(
                            label="📥 Télécharger la musique",
                            data=f.read(),
                            file_name=f"reve_musique_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
                            mime="audio/wav",
                            use_container_width=True
                        )
                else:
                    st.warning("⚠️ Impossible de créer la musique")
        else:
            progress_bar.progress(90)
    
    # Finalisation
    progress_bar.progress(100)
    status_text.text("✅ Traitement terminé!")
    
    # Résumé avec prompts utilisés
    st.markdown("---")
    st.subheader("✨ Votre création multimédia")
    
    col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
    
    with col_summary1:
        if emotions:
            dominant_emotion = max(emotions, key=emotions.get)
            st.metric("🎭 Émotion", dominant_emotion.capitalize())
    
    with col_summary2:
        media_count = sum([1 for x in [image, video_path, music_path] if x])
        st.metric("🎨 Médias", f"{media_count}/3")
    
    with col_summary3:
        word_count = len(dream_text.split())
        st.metric("📝 Mots", word_count)
    
    with col_summary4:
        st.metric("⏱️ Temps", "~1-2 min")
    
    # Affichage de tous les prompts générés
    if generated_prompt:
        with st.expander("📋 Récapitulatif des prompts générés", expanded=False):
            st.markdown("**🖼️ Prompt Image :**")
            st.code(generated_prompt, language="text")
            
            if generate_video:
                video_prompt = synthesizer.generate_video_prompt(dream_text)
                st.markdown("**🎬 Prompt Vidéo :**")
                st.code(video_prompt, language="text")
            
            if generate_music:
                music_prompt = synthesizer.generate_music_prompt(dream_text)
                st.markdown("**🎵 Prompt Musical :**")
                st.code(music_prompt, language="text")
    
    # Bouton nouveau rêve
    if st.button("🆕 Créer un nouveau rêve", type="secondary"):
        st.rerun()


def handle_dream_history(synthesizer) -> None:
    """
    Gère l'affichage de l'historique des rêves avec filtres et recherche.
    
    Args:
        synthesizer: Instance du synthétiseur
    """
    logger.debug("📚 Affichage de la page historique des rêves")
    
    st.header("📚 Historique de vos rêves")
    st.markdown("Retrouvez et explorez tous vos rêves sauvegardés")
    
    # Chargement des rêves
    dreams = synthesizer.load_dreams()
    
    if not dreams:
        # Aucun rêve trouvé
        st.info("🌙 **Aucun rêve sauvegardé pour l'instant.**")
        st.markdown("""
        💡 **Suggestions :**
        - Commencez par créer votre premier rêve dans l'onglet "🎤 Nouveau rêve"
        - Racontez vos rêves récents ou passés
        - Explorez les différentes émotions et images générées
        """)
        
        if st.button("🆕 Créer mon premier rêve", type="primary"):
            st.session_state.page = "🎤 Nouveau rêve"
            st.rerun()
        
        return
    
    # Statistiques générales
    st.success(f"📊 **{len(dreams)} rêve(s) trouvé(s)** dans votre collection")
    
    # Statistiques détaillées dans des métriques
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    with col_stats1:
        total_words = sum(len(dream.get("text", "").split()) for dream in dreams)
        st.metric("📝 Mots totaux", f"{total_words:,}")
    
    with col_stats2:
        dreams_with_images = len([d for d in dreams if d.get("image_path")])
        st.metric("🖼️ Avec images", f"{dreams_with_images}/{len(dreams)}")
    
    with col_stats3:
        # Calcul de l'émotion la plus fréquente
        all_emotions = {}
        for dream in dreams:
            if dream.get("emotions"):
                dominant = max(dream["emotions"], key=dream["emotions"].get)
                all_emotions[dominant] = all_emotions.get(dominant, 0) + 1
        
        if all_emotions:
            most_common = max(all_emotions, key=all_emotions.get)
            st.metric("🎭 Émotion fréquente", most_common.capitalize())
        else:
            st.metric("🎭 Émotions", "N/A")
    
    with col_stats4:
        # Rêve le plus récent
        if dreams:
            try:
                latest_date = max(dreams, key=lambda x: x.get("date", ""))["date"]
                days_ago = (datetime.now() - datetime.fromisoformat(latest_date)).days
                st.metric("📅 Dernier rêve", f"Il y a {days_ago}j")
            except:
                st.metric("📅 Dernier rêve", "Récent")
    
    # Filtres et recherche
    st.markdown("---")
    st.subheader("🔍 Filtres et recherche")
    
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        # Filtre par émotion dominante
        all_emotions = set()
        for dream in dreams:
            if dream.get("emotions"):
                all_emotions.update(dream["emotions"].keys())
        
        emotion_options = ["Toutes"] + sorted(list(all_emotions))
        selected_emotion = st.selectbox(
            "🎭 Émotion dominante:",
            emotion_options,
            help="Filtrer par l'émotion la plus forte du rêve"
        )
    
    with col_filter2:
        # Filtre par période
        date_filter = st.selectbox(
            "📅 Période:",
            ["Toutes", "Dernière semaine", "Dernier mois", "Dernière année"],
            help="Filtrer par date de création du rêve"
        )
    
    with col_filter3:
        # Recherche par texte
        search_text = st.text_input(
            "🔍 Rechercher:",
            placeholder="Mots-clés dans le titre, texte ou tags...",
            help="Recherche dans le titre, contenu et tags des rêves"
        )
    
    # Application des filtres
    filtered_dreams = filter_dreams(dreams, selected_emotion, date_filter, search_text)
    
    # Affichage des résultats filtrés
    st.markdown("---")
    
    if filtered_dreams:
        # Tri par date (plus récents en premier)
        try:
            filtered_dreams.sort(key=lambda x: x.get("date", ""), reverse=True)
        except:
            logger.warning("⚠️ Problème de tri par date")
        
        # En-tête des résultats
        result_count = len(filtered_dreams)
        total_count = len(dreams)
        
        if result_count == total_count:
            st.subheader(f"📝 Tous vos rêves ({result_count})")
        else:
            st.subheader(f"📝 Résultats filtrés ({result_count}/{total_count})")
        
        # Options d'affichage
        col_display1, col_display2 = st.columns([3, 1])
        
        with col_display1:
            # Boutons d'action groupée
            if st.button("📊 Analyser les tendances émotionnelles"):
                show_emotion_trends(filtered_dreams)
        
        with col_display2:
            # Tri des résultats
            sort_option = st.selectbox(
                "Trier par:",
                ["Date (récent)", "Date (ancien)", "Titre", "Émotions"],
                help="Choisir l'ordre d'affichage"
            )
            
            if sort_option == "Date (ancien)":
                filtered_dreams.sort(key=lambda x: x.get("date", ""))
            elif sort_option == "Titre":
                filtered_dreams.sort(key=lambda x: x.get("title", ""))
            elif sort_option == "Émotions":
                filtered_dreams.sort(key=lambda x: max(x.get("emotions", {}).values()) if x.get("emotions") else 0, reverse=True)
        
        # Affichage des cartes de rêves
        for dream in filtered_dreams:
            display_dream_card(dream)
            
        logger.info(f"✅ Affichage de {len(filtered_dreams)} rêves terminé")
        
    else:
        # Aucun résultat trouvé
        st.warning("🔍 **Aucun rêve ne correspond à vos critères de recherche.**")
        
        st.markdown("""
        💡 **Suggestions :**
        - Modifiez les filtres ci-dessus
        - Essayez une recherche plus large
        - Vérifiez l'orthographe des mots-clés
        """)
        
        if st.button("🔄 Réinitialiser les filtres"):
            st.rerun()


def show_emotion_trends(dreams: List[Dict[str, Any]]) -> None:
    """
    Affiche une analyse des tendances émotionnelles des rêves.
    
    Args:
        dreams (List[Dict[str, Any]]): Liste des rêves à analyser
    """
    logger.debug(f"📈 Analyse des tendances émotionnelles pour {len(dreams)} rêves")
    
    if not dreams:
        st.warning("Aucun rêve à analyser")
        return
    
    st.markdown("---")
    st.subheader("📈 Analyse des tendances émotionnelles")
    
    # Collecte des données émotionnelles
    emotion_data = {}
    date_emotion_data = []
    
    for dream in dreams:
        emotions = dream.get("emotions", {})
        dream_date = dream.get("date", "")
        
        if emotions:
            # Agrégation globale
            for emotion, score in emotions.items():
                if emotion not in emotion_data:
                    emotion_data[emotion] = []
                emotion_data[emotion].append(score)
            
            # Données temporelles
            try:
                date_obj = datetime.fromisoformat(dream_date)
                dominant_emotion = max(emotions, key=emotions.get)
                date_emotion_data.append({
                    "date": date_obj,
                    "emotion": dominant_emotion,
                    "score": emotions[dominant_emotion]
                })
            except:
                continue
    
    if emotion_data:
        col_trend1, col_trend2 = st.columns(2)
        
        with col_trend1:
            # Graphique des moyennes émotionnelles
            st.markdown("#### 🎭 Moyennes émotionnelles")
            
            avg_emotions = {emotion: sum(scores)/len(scores) for emotion, scores in emotion_data.items()}
            avg_df = pd.DataFrame(list(avg_emotions.items()), columns=["Émotion", "Score moyen"])
            avg_df = avg_df.sort_values("Score moyen", ascending=True)
            
            st.bar_chart(avg_df.set_index("Émotion")["Score moyen"], height=300)
        
        with col_trend2:
            # Statistiques détaillées
            st.markdown("#### 📊 Statistiques détaillées")
            
            for emotion, scores in sorted(emotion_data.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True):
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
                occurrences = len([s for s in scores if s > 0.3])  # Seuil significatif
                
                st.write(f"**{emotion.capitalize()}**")
                st.write(f"• Moyenne: {avg_score:.2f}")
                st.write(f"• Maximum: {max_score:.2f}")
                st.write(f"• Occurrences significatives: {occurrences}")
                st.write("")
        
        # Évolution temporelle si suffisamment de données
        if len(date_emotion_data) > 3:
            st.markdown("#### 📅 Évolution temporelle")
            
            # Groupement par semaine pour lisibilité
            weekly_data = {}
            for entry in date_emotion_data:
                week_key = entry["date"].strftime("%Y-W%U")
                if week_key not in weekly_data:
                    weekly_data[week_key] = {"emotions": {}, "count": 0}
                
                emotion = entry["emotion"]
                if emotion not in weekly_data[week_key]["emotions"]:
                    weekly_data[week_key]["emotions"][emotion] = 0
                weekly_data[week_key]["emotions"][emotion] += 1
                weekly_data[week_key]["count"] += 1
            
            # Affichage simplifié de l'évolution
            st.write("**Émotions dominantes par période:**")
            for week, data in sorted(weekly_data.items())[-8:]:  # 8 dernières semaines
                dominant = max(data["emotions"], key=data["emotions"].get)
                count = data["count"]
                st.write(f"• Semaine {week}: **{dominant}** ({count} rêve(s))")


def handle_configuration() -> None:
    """Gère la page de configuration de l'application."""
    logger.debug("⚙️ Affichage de la page configuration")
    
    st.header("⚙️ Configuration")
    st.markdown("Personnalisez votre expérience du Synthétiseur de rêves")
    
    # Configuration des APIs
    st.subheader("🔑 Configuration des APIs")
    
    with st.expander("🛠️ État des services", expanded=True):
        col_api1, col_api2, col_api3 = st.columns(3)
        
        with col_api1:
            groq_status = "✅ Configuré" if st.secrets.get("GROQ_API_KEY") else "❌ Manquant"
            st.metric("🚀 Groq API", groq_status)
            st.caption("Génération de prompts")
        
        with col_api2:
            mistral_status = "✅ Configuré" if st.secrets.get("MISTRAL_API_KEY") else "❌ Manquant"
            st.metric("🧠 Mistral AI", mistral_status)
            st.caption("Analyse émotionnelle")
        
        with col_api3:
            clipdrop_status = "✅ Configuré" if st.secrets.get("CLIPDROP_API_KEY") else "❌ Manquant"
            st.metric("🎨 ClipDrop", clipdrop_status)
            st.caption("Génération d'images")
    
    # Instructions de configuration
    if not all([st.secrets.get("GROQ_API_KEY"), st.secrets.get("MISTRAL_API_KEY"), st.secrets.get("CLIPDROP_API_KEY")]):
        st.warning("⚠️ **Certaines clés API sont manquantes**")
        
        st.markdown("""
        **Pour configurer les APIs :**
        
        1. **Créez vos comptes** sur les plateformes :
           - [Groq](https://groq.com/) - Pour la génération de prompts
           - [Mistral AI](https://console.mistral.ai/) - Pour l'analyse émotionnelle  
           - [ClipDrop](https://clipdrop.co/apis) - Pour la génération d'images
        
        2. **Récupérez vos clés API** depuis les tableaux de bord
        
        3. **Ajoutez-les** dans le fichier `.streamlit/secrets.toml` :
        ```toml
        GROQ_API_KEY = "votre_clé_groq"
        MISTRAL_API_KEY = "votre_clé_mistral"
        CLIPDROP_API_KEY = "votre_clé_clipdrop"
        ```
        
        4. **Redémarrez** l'application
        """)
    
    # Paramètres de l'application
    st.markdown("---")
    st.subheader("🎛️ Paramètres de l'application")
    
    # Paramètres de génération
    with st.expander("🎨 Paramètres de génération"):
        st.slider(
            "🎯 Créativité des prompts",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Plus élevé = plus créatif mais moins précis"
        )
        
        st.slider(
            "🧠 Précision de l'analyse émotionnelle",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Plus bas = plus déterministe"
        )
        
        st.selectbox(
            "🗣️ Langue de transcription",
            ["fr-FR", "en-US", "es-ES", "it-IT", "de-DE"],
            help="Langue pour la reconnaissance vocale"
        )
    
    # Gestion des données
    with st.expander("💾 Gestion des données"):
        dreams_file = Path("dreams_data.json")
        if dreams_file.exists():
            try:
                with open(dreams_file, "r", encoding="utf-8") as f:
                    dreams_data = json.load(f)
                    dreams_count = len(dreams_data)
            except:
                dreams_count = 0
        else:
            dreams_count = 0
        
        st.write(f"📊 **{dreams_count} rêves** actuellement sauvegardés")
        
        col_data1, col_data2 = st.columns(2)
        
        with col_data1:
            if st.button("📥 Exporter mes rêves", help="Télécharger tous vos rêves en JSON"):
                export_dreams_data()
        
        with col_data2:
            uploaded_file = st.file_uploader("📤 Importer des rêves", type="json", help="Importer un fichier de sauvegarde")
            if uploaded_file:
                import_dreams_data(uploaded_file)
    
    # Informations système
    st.markdown("---")
    st.subheader("ℹ️ Informations système")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.write("**🔧 Version de l'application**")
        st.code("1.0.0")
        
        st.write("**📅 Dernière mise à jour**")
        st.code("2025-07-03")
    
    with col_info2:
        st.write("**📊 Stockage utilisé**")
        try:
            storage_size = sum(f.stat().st_size for f in Path(".").rglob("*") if f.is_file()) / 1024 / 1024
            st.code(f"{storage_size:.1f} MB")
        except:
            st.code("N/A")
        
        st.write("**🌐 Statut réseau**")
        st.code("🟢 Connecté")


def export_dreams_data() -> None:
    """Exporte les données des rêves pour téléchargement."""
    logger.debug("📥 Export des données de rêves")
    
    try:
        dreams_file = Path("dreams_data.json")
        if dreams_file.exists():
            with open(dreams_file, "r", encoding="utf-8") as f:
                dreams_data = f.read()
            
            st.download_button(
                label="📥 Télécharger mes rêves",
                data=dreams_data,
                file_name=f"mes_reves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            st.success("✅ Export préparé ! Cliquez sur le bouton pour télécharger.")
        else:
            st.warning("⚠️ Aucun rêve à exporter")
    except Exception as e:
        st.error(f"❌ Erreur lors de l'export: {e}")
        logger.error(f"❌ Erreur export: {e}")


def import_dreams_data(uploaded_file) -> None:
    """Importe des données de rêves depuis un fichier."""
    logger.debug("📤 Import des données de rêves")
    
    try:
        # Lecture du fichier uploadé
        imported_data = json.loads(uploaded_file.read().decode("utf-8"))
        
        if not isinstance(imported_data, list):
            st.error("❌ Format de fichier invalide")
            return
        
        # Validation basique des données
        valid_dreams = []
        for dream in imported_data:
            if isinstance(dream, dict) and "text" in dream:
                valid_dreams.append(dream)
        
        if valid_dreams:
            # Chargement des rêves existants
            dreams_file = Path("dreams_data.json")
            existing_dreams = []
            
            if dreams_file.exists():
                with open(dreams_file, "r", encoding="utf-8") as f:
                    existing_dreams = json.load(f)
            
            # Fusion des données
            all_dreams = existing_dreams + valid_dreams
            
            # Sauvegarde
            with open(dreams_file, "w", encoding="utf-8") as f:
                json.dump(all_dreams, f, ensure_ascii=False, indent=2)
            
            st.success(f"✅ {len(valid_dreams)} rêve(s) importé(s) avec succès !")
            st.rerun()
        else:
            st.error("❌ Aucun rêve valide trouvé dans le fichier")
            
    except json.JSONDecodeError:
        st.error("❌ Fichier JSON invalide")
    except Exception as e:
        st.error(f"❌ Erreur lors de l'import: {e}")
        logger.error(f"❌ Erreur import: {e}")


# ================================
# FONCTION PRINCIPALE
# ================================

def main() -> None:
    """Fonction principale de l'application Streamlit."""
    logger.info("🚀 Démarrage de l'application Synthétiseur de rêves")
    
    # Configuration de la page Streamlit
    st.set_page_config(
        page_title="🌙 Synthétiseur de rêves",
        page_icon="🌙",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/votre-repo/dream-synthesizer',
            'Report a bug': 'https://github.com/votre-repo/dream-synthesizer/issues',
            'About': """
            # Synthétiseur de rêves 🌙
            
            Transformez vos rêves en art avec l'intelligence artificielle !
            
            **Version:** 1.0.0  
            **Date:** 2025-07-03
            """
        }
    )
    
    # Titre principal avec style
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #FFD700; font-size: 3em;">🌙 Synthétiseur de rêves</h1>
        <p style="font-size: 1.2em; color: #B8B8B8;">Transformez vos rêves en images artistiques avec l'IA</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Présentation de l'application
    with st.expander("ℹ️ Comment ça fonctionne", expanded=False):
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.markdown("""
            ### 🎤 1. Racontez
            - Tapez votre rêve ou uploadez un audio
            - Décrivez les détails, émotions, couleurs
            - Plus c'est détaillé, mieux c'est !
            """)
        
        with col_info2:
            st.markdown("""
            ### 🧠 2. Analysez
            - L'IA analyse vos émotions
            - Génère un prompt artistique optimisé
            - Traite le tout en quelques secondes
            """)
        
        with col_info3:
            st.markdown("""
            ### 🎨 3. Visualisez
            - Recevez une image unique de votre rêve
            - Consultez l'analyse émotionnelle
            - Sauvegardez dans votre collection
            """)
    
    # Initialisation du synthétiseur
    try:
        # Import de la classe DreamSynthesizer depuis le fichier principal
        
        synthesizer = DreamSynthesizer()
        logger.info("✅ Synthétiseur initialisé avec succès")
        
    except Exception as e:
        st.error(f"❌ **Erreur d'initialisation:** {e}")
        st.markdown("""
        **Causes possibles :**
        - Clés API manquantes ou invalides
        - Problème de connexion réseau
        - Configuration incorrecte
        
        Consultez la page **Configuration** pour résoudre le problème.
        """)
        logger.error(f"❌ Erreur d'initialisation: {e}")
        return
    
    # Navigation dans la sidebar
    st.sidebar.title("📋 Navigation")
    st.sidebar.markdown("Choisissez une action ci-dessous :")
    
    # Gestion de l'état de la page
    if "page" not in st.session_state:
        st.session_state.page = "🎤 Nouveau rêve"
    
    # Menu de navigation
    page = st.sidebar.radio(
        "Sections:",
        ["🎤 Nouveau rêve", "📚 Historique", "⚙️ Configuration"],
        index=["🎤 Nouveau rêve", "📚 Historique", "⚙️ Configuration"].index(st.session_state.page)
    )
    
    # Mise à jour de l'état si changement
    if page != st.session_state.page:
        st.session_state.page = page
    
    # Informations dans la sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Statistiques")
    
    try:
        dreams = synthesizer.load_dreams()
        st.sidebar.metric("💭 Rêves totaux", len(dreams))
        
        if dreams:
            dreams_with_images = len([d for d in dreams if d.get("image_path")])
            st.sidebar.metric("🖼️ Avec images", dreams_with_images)
            
            # Dernière activité
            try:
                latest = max(dreams, key=lambda x: x.get("date", ""))["date"]
                days_ago = (datetime.now() - datetime.fromisoformat(latest)).days
                st.sidebar.metric("📅 Dernière activité", f"Il y a {days_ago}j")
            except:
                st.sidebar.metric("📅 Dernière activité", "Récente")
                
    except Exception as e:
        logger.warning(f"⚠️ Erreur statistiques sidebar: {e}")
    
    # Liens utiles
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 Liens utiles")
    st.sidebar.markdown("""
    - 📖 [Documentation](https://github.com/votre-repo/dream-synthesizer)
    - 🐛 [Signaler un bug](https://github.com/votre-repo/dream-synthesizer/issues)
    - 💡 [Suggestions](https://github.com/votre-repo/dream-synthesizer/discussions)
    """)
    
    # Gestion des pages
    logger.debug(f"📄 Affichage de la page: {page}")
    
    if page == "🎤 Nouveau rêve":
        handle_new_dream(synthesizer)
        
    elif page == "📚 Historique":
        handle_dream_history(synthesizer)
        
    elif page == "⚙️ Configuration":
        handle_configuration()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🌙 Synthétiseur de rêves • Version 1.0.0 • Créé avec ❤️ et Streamlit</p>
        <p style="font-size: 0.8em;">Transformez vos rêves en art • Explorez votre inconscient • Créez des souvenirs visuels</p>
    </div>
    """, unsafe_allow_html=True)


# ================================
# POINT D'ENTRÉE DE L'APPLICATION
# ================================

if __name__ == "__main__":
    # Création du dossier logs si nécessaire
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("="*50)
    logger.info("🌙 SYNTHÉTISEUR DE RÊVES - DÉMARRAGE")
    logger.info("="*50)
    
    try:
        main()
        logger.info("✅ Application exécutée avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur fatale dans l'application: {e}")
        st.error(f"❌ **Erreur critique:** {e}")
        st.markdown("Consultez les logs pour plus de détails.")
    finally:
        logger.info("🔚 Fin de l'exécution de l'application")