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
    Classe principale pour le synthétiseur de rêves.
    
    Cette classe gère toutes les fonctionnalités principales :
    - Transcription audio vers texte
    - Génération d'images à partir du texte
    - Analyse émotionnelle
    - Stockage et récupération des rêves
    """
    
    def __init__(self):
        """Initialise le synthétiseur avec les configurations nécessaires."""
        logger.debug("Initialisation du DreamSynthesizer")
        
        # Configuration des APIs (à remplacer par vos vraies clés)
        self.groq_api_key = st.secrets.get("GROQ_API_KEY", "")
        self.clipdrop_api_key = st.secrets.get("CLIPDROP_API_KEY", "")
        self.mistral_api_key = st.secrets.get("MISTRAL_API_KEY", "")
        
        # URLs des APIs
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.clipdrop_url = "https://clipdrop-api.co/text-to-image/v1"
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
        
        Args:
            dream_text: Texte du rêve à analyser
            
        Returns:
            Dict: Scores émotionnels pour chaque catégorie
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
        
        Args:
            dream_text: Texte du rêve
            
        Returns:
            str: Prompt optimisé pour la génération d'image
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
                "max_tokens": 150
            }
            
            logger.debug("Envoi de la requête à l'API Groq pour le prompt")
            response = requests.post(self.groq_url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                image_prompt = result['choices'][0]['message']['content']
                logger.debug(f"Prompt généré: {image_prompt}")
                return image_prompt
            else:
                logger.error(f"Erreur API Groq: {response.status_code}")
                return f"Dreamlike scene: {dream_text[:100]}"
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération du prompt: {e}")
            return f"Dreamlike scene: {dream_text[:100]}"
    
    def generate_image(self, dream_text: str) -> Optional[Image.Image]:
        """
        Génère une image à partir du texte du rêve via l'API ClipDrop.
        Avec fallback vers une alternative si ClipDrop échoue.
        """
        logger.debug(f"🖼️ Génération d'image pour: {dream_text[:50]}...")
        
        if not self.clipdrop_api_key:
            logger.warning("⚠️ Clé API ClipDrop manquante, utilisation du mode placeholder")
            return self.create_placeholder_image(dream_text)
        
        try:
            # Génération du prompt optimisé
            image_prompt = self.generate_image_prompt(dream_text)
            
            # Configuration de la requête API ClipDrop
            headers = {
                "x-api-key": self.clipdrop_api_key,
            }
            
            data = {
                "prompt": image_prompt
            }
            
            logger.debug(f"📡 Envoi de la requête à ClipDrop avec le prompt: {image_prompt[:100]}...")
            start_time = datetime.now()
            
            response = requests.post(
                self.clipdrop_url, 
                headers=headers, 
                data=data,
                timeout=60  # Plus long pour la génération d'image
            )
            
            response_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"⏱️ Temps de réponse API ClipDrop: {response_time:.2f}s")
            
            if response.status_code == 200:
                logger.info("✅ Image générée avec succès via ClipDrop")
                
                # Conversion de la réponse en image PIL
                image = Image.open(io.BytesIO(response.content))
                
                logger.debug(f"🖼️ Image créée: {image.size} pixels, mode {image.mode}")
                return image
                
            else:
                logger.error(f"❌ Erreur API ClipDrop: {response.status_code} - {response.text}")
                logger.info("🔄 Fallback vers image alternative")
                return self.generate_image_alternative(dream_text)
                
        except requests.exceptions.Timeout:
            logger.error("❌ Timeout lors de la génération d'image (>60s)")
            return self.generate_image_alternative(dream_text)
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur de requête à l'API ClipDrop: {e}")
            return self.generate_image_alternative(dream_text)
        except Exception as e:
            logger.error(f"❌ Erreur inattendue lors de la génération d'image: {e}")
            return self.generate_image_alternative(dream_text)
        
    def generate_image_alternative(self, dream_text: str) -> Optional[Image.Image]:
        """
        Fallback image generation using Hugging Face Stable Diffusion API.
        """
        try:
            image_prompt = self.generate_image_prompt(dream_text)
            hf_token = st.secrets.get("HF_TOKEN", "")
            if not hf_token:
                logger.warning("⚠️ Aucun token Hugging Face trouvé dans secrets")
                raise Exception("Missing Hugging Face token")

            logger.debug("📡 Envoi à Hugging Face (Stable Diffusion)")
            response = requests.post(
                "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2",
                headers={"Authorization": f"Bearer {hf_token}"},
                json={"inputs": image_prompt},
                timeout=60
            )

            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                logger.info("✅ Image générée avec Hugging Face (Stable Diffusion)")
                return image
            else:
                logger.warning(f"⚠️ Erreur Hugging Face: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"❌ Erreur Hugging Face: {e}")

        # Fallback final : placeholder
        logger.debug("🎨 Fallback final : génération d'une image placeholder")
        return self.create_placeholder_image(dream_text)




    def create_placeholder_image(self, dream_text: str) -> Image.Image:
        """
        Crée une image placeholder artistique basée sur le texte du rêve.
        """
        from PIL import Image, ImageDraw, ImageFont
        import random
        
        # Créer une image avec dégradé
        width, height = 512, 512
        image = Image.new('RGB', (width, height), color=(30, 30, 60))
        draw = ImageDraw.Draw(image)
        
        # Dégradé de couleur basé sur le rêve
        colors = [
            (138, 43, 226),   # Purple
            (75, 0, 130),     # Indigo  
            (255, 215, 0),    # Gold
            (47, 79, 79),     # Dark slate gray
            (25, 25, 112)     # Midnight blue
        ]
        
        # Choisir des couleurs basées sur le hash du texte
        text_hash = hash(dream_text) % len(colors)
        primary_color = colors[text_hash]
        secondary_color = colors[(text_hash + 1) % len(colors)]
        
        # Créer un dégradé circulaire
        for i in range(min(width, height) // 2):
            factor = i / (min(width, height) // 2)
            r = int(primary_color[0] * (1 - factor) + secondary_color[0] * factor)
            g = int(primary_color[1] * (1 - factor) + secondary_color[1] * factor)
            b = int(primary_color[2] * (1 - factor) + secondary_color[2] * factor)
            
            draw.ellipse([
                width//2 - i, height//2 - i,
                width//2 + i, height//2 + i
            ], outline=(r, g, b))
        
        # Ajouter des formes géométriques inspirées du rêve
        words = dream_text.lower().split()
        
        for i, word in enumerate(words[:5]):  # Maximum 5 formes
            x = (hash(word) % (width - 100)) + 50
            y = (hash(word + str(i)) % (height - 100)) + 50
            size = 30 + (len(word) * 10) % 50
            
            color = colors[hash(word) % len(colors)]
            alpha = 100  # Transparence
            
            # Différentes formes selon le mot
            if len(word) % 3 == 0:
                # Cercle
                draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], 
                            fill=(*color, alpha))
            elif len(word) % 3 == 1:
                # Triangle (approximé avec un polygone)
                points = [
                    (x, y-size//2),
                    (x-size//2, y+size//2),
                    (x+size//2, y+size//2)
                ]
                draw.polygon(points, fill=(*color, alpha))
            else:
                # Rectangle
                draw.rectangle([x-size//2, y-size//2, x+size//2, y+size//2], 
                            fill=(*color, alpha))
        
        # Ajouter le titre du rêve en bas
        try:
            # Essayer d'utiliser une police système
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Texte avec le début du rêve
        preview_text = dream_text[:30] + "..." if len(dream_text) > 30 else dream_text
        text_color = (255, 255, 255, 200)  # Blanc semi-transparent
        
        # Centrer le texte
        bbox = draw.textbbox((0, 0), preview_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (width - text_width) // 2
        text_y = height - 60
        
        # Fond semi-transparent pour le texte
        draw.rectangle([text_x-10, text_y-5, text_x+text_width+10, text_y+30], 
                    fill=(0, 0, 0, 128))
        
        draw.text((text_x, text_y), preview_text, font=font, fill=text_color)
        
        # Ajouter watermark
        watermark = "🌙 Synthétiseur de rêves"
        draw.text((10, 10), watermark, font=font, fill=(255, 255, 255, 150))
        
        logger.info("✅ Image placeholder créée")
        return image
    
    def save_dream(self, dream_data: Dict[str, Any]) -> bool:
        """
        Sauvegarde un rêve dans le stockage local (fichier JSON).
        
        Args:
            dream_data (Dict[str, Any]): Données du rêve à sauvegarder
            
        Returns:
            bool: True si sauvegarde réussie, False sinon
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
        
        Returns:
            List[Dict[str, Any]]: Liste des rêves chargés
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
        
        Args:
            image (Image.Image): Image à sauvegarder
            dream_id (str): Identifiant unique du rêve
            
        Returns:
            Optional[str]: Chemin du fichier sauvegardé ou None si erreur
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
                    st.image(image, caption="Image du rêve", use_column_width=True)
                    
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
    Traite un rêve complet : analyse émotionnelle + génération d'image.
    
    Args:
        synthesizer: Instance du synthétiseur
        dream_text (str): Texte du rêve à traiter
    """
    logger.info(f"🔄 Début du traitement complet du rêve: {dream_text[:50]}...")
    
    # Initialisation des variables
    emotions = {}
    image = None
    
    # Interface de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Colonnes pour l'affichage des résultats
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎭 Analyse émotionnelle")
        emotion_placeholder = st.empty()
        
        # Étape 1: Analyse des émotions
        with st.spinner("🧠 Analyse des émotions en cours..."):
            status_text.text("🔍 Analyse émotionnelle...")
            progress_bar.progress(25)
            
            logger.debug("🎭 Début de l'analyse émotionnelle")
            emotions = synthesizer.analyze_emotion(dream_text)
            
            if emotions:
                emotion_placeholder.success("✅ Analyse émotionnelle terminée!")
                create_emotion_chart(emotions)
                logger.info("✅ Analyse émotionnelle réussie")
            else:
                emotion_placeholder.error("❌ Erreur lors de l'analyse émotionnelle")
                logger.error("❌ Échec de l'analyse émotionnelle")
    
    with col2:
        st.subheader("🎨 Génération d'image")
        image_placeholder = st.empty()
        
        # Étape 2: Génération de l'image
        with st.spinner("🖼️ Création de l'image artistique..."):
            status_text.text("🎨 Génération de l'image...")
            progress_bar.progress(75)
            
            logger.debug("🖼️ Début de la génération d'image")
            image_prompt = synthesizer.generate_image_prompt(dream_text)
            st.markdown("**📝 Prompt artistique généré :**")
            st.code(image_prompt, language="markdown")

            image = synthesizer.generate_image(dream_text)
            
            if image:
                image_placeholder.success("✅ Image générée avec succès!")
                st.image(image, caption="🌙 Votre rêve visualisé", use_column_width=True)
                
                # Bouton de téléchargement immédiat
                img_buffer = io.BytesIO()
                image.save(img_buffer, format="PNG")
                st.download_button(
                    label="📥 Télécharger l'image",
                    data=img_buffer.getvalue(),
                    file_name=f"reve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                logger.info("✅ Génération d'image réussie")
            else:
                image_placeholder.error("❌ Erreur lors de la génération d'image")
                logger.error("❌ Échec de la génération d'image")
    
    # Finalisation
    progress_bar.progress(100)
    status_text.text("✅ Traitement terminé!")
    
    # Section de sauvegarde
    st.markdown("---")
    st.subheader("💾 Sauvegarder ce rêve")
    
    # Formulaire de sauvegarde
    with st.form("save_dream_form", clear_on_submit=True):
        col_save1, col_save2 = st.columns([2, 1])
        
        with col_save1:
            dream_title = st.text_input(
                "🏷️ Titre du rêve:",
                value=f"Rêve du {datetime.now().strftime('%d/%m/%Y')}",
                help="Donnez un titre mémorable à votre rêve"
            )
            
            dream_tags = st.text_input(
                "🏷️ Tags (séparés par des virgules):",
                placeholder="fantastique, aventure, couleurs vives, famille...",
                help="Ajoutez des mots-clés pour retrouver facilement ce rêve"
            )
        
        with col_save2:
            st.markdown("**📊 Résumé:**")
            st.write(f"📝 {len(dream_text.split())} mots")
            if emotions:
                dominant_emotion = max(emotions, key=emotions.get)
                st.write(f"🎭 Émotion dominante: **{dominant_emotion}**")
            if image:
                st.write(f"🖼️ Image: {image.size[0]}x{image.size[1]}px")
        
        # Bouton de soumission
        submitted = st.form_submit_button("💾 **Sauvegarder le rêve**", type="primary", use_container_width=True)
        
        if submitted:
            # Création de l'ID unique du rêve
            dream_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Préparation des données du rêve
            dream_data = {
                "id": dream_id,
                "title": dream_title.strip() if dream_title.strip() else f"Rêve du {datetime.now().strftime('%d/%m/%Y')}",
                "text": dream_text.strip(),
                "emotions": emotions,
                "tags": [tag.strip() for tag in dream_tags.split(",") if tag.strip()],
                "date": datetime.now().isoformat(),
                "image_path": None
            }
            
            # Sauvegarde de l'image si disponible
            if image:
                logger.debug(f"💾 Sauvegarde de l'image pour le rêve {dream_id}")
                image_path = synthesizer.save_dream_image(image, dream_id)
                if image_path:
                    dream_data["image_path"] = image_path
                    logger.info(f"✅ Image sauvegardée: {image_path}")
            
            # Sauvegarde du rêve complet
            logger.debug(f"💾 Sauvegarde des données du rêve {dream_id}")
            if synthesizer.save_dream(dream_data):
                st.success("🎉 **Rêve sauvegardé avec succès!**")
                st.balloons()  # Animation de célébration
                logger.info(f"✅ Rêve {dream_id} sauvegardé avec succès")
                
                # Proposition de créer un nouveau rêve
                if st.button("🆕 Créer un nouveau rêve", type="secondary"):
                    st.rerun()
            else:
                st.error("❌ Erreur lors de la sauvegarde du rêve")
                logger.error(f"❌ Échec de sauvegarde du rêve {dream_id}")


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