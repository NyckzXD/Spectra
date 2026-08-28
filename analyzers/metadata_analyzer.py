import re
import traceback
from PIL import Image
from PIL.ExifTags import TAGS

# Campos que contêm texto livre digitado por humanos (legendas, descrições, autoria).
# Palavras de gerador de IA que caem aqui NÃO devem contar como assinatura, pois
# termos como "runway", "muse", "imagen", "gemini", "sora" são palavras/nomes comuns
# do dia a dia e geram falsos positivos severos se buscados em texto livre.
FREE_TEXT_FIELDS = {
    'imagedescription', 'usercomment', 'artist', 'copyright', 'comment',
    'xpcomment', 'xpsubject', 'xptitle', 'title', 'subject'
}

# Termos inequívocos: nomes de produto/engine compostos ou com pontuação específica.
# Praticamente não aparecem por acaso em texto comum -> seguros para scan em qualquer campo.
UNAMBIGUOUS_AI_TAGS = [
    'stable diffusion', 'dall-e', 'dall·e', 'midjourney', 'comfyui', 'automatic1111',
    'a1111', 'novelai', 'adobe firefly', 'flux.1', 'invoke ai', 'invokeai',
    'foocus', 'fooocus', 'dreamstudio', 'dream studio',
    'craiyon', 'leonardo.ai', 'leonardo ai', 'playground ai', 'ideogram',
    'copilot designer', 'bing image creator', 'nightcafe', 'artbreeder',
    'kling ai', 'recraft ai',
    'steps:', 'sampler:', 'cfg scale:', 'seed:', 'negative prompt:',
    'lora:', 'embedding:', 'vae:', 'clip skip'
]

# Termos ambíguos: palavras/nomes comuns que só devem contar quando aparecem em
# campos TÉCNICOS/ESTRUTURAIS (Software, Make, Model, parameters, workflow),
# nunca em campos de texto livre digitados por humanos.
AMBIGUOUS_AI_TAGS = [
    'runway', 'pika', 'sora', 'kling', 'recraft', 'grok', 'xai',
    'gemini', 'imagen', 'parti', 'muse', 'flux',
]


def _build_scan_string(metadata: dict) -> str:
    """Concatena apenas campos técnicos/estruturais, excluindo texto livre do usuário."""
    parts = []
    for k, v in metadata.items():
        if str(k).lower() not in FREE_TEXT_FIELDS:
            parts.append(f"{k}: {v}")
    return " | ".join(parts).lower()


def _word_match(term: str, text: str) -> bool:
    """Casamento por palavra/frase inteira (evita 'lg' casando dentro de 'light')."""
    pattern = r'(?<![a-z0-9])' + re.escape(term) + r'(?![a-z0-9])'
    return re.search(pattern, text) is not None


def analyze_metadata(image_path: str) -> dict:
    """
    Analyzes image metadata to detect AI generation signatures or authentic camera origins.
    Checks EXIF, PNG tEXt chunks, IPTC/XMP markers, C2PA, and known AI generator tags.
    Returns: {'score': int, 'has_signal': bool, 'findings': [str], 'details': {'metadata': {key: value}}}
    """
    try:
        img = Image.open(image_path)
        metadata = {}
        info = img.getexif()

        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                try:
                    metadata[decoded] = str(value)[:500]
                except Exception:
                    metadata[decoded] = '<unreadable>'

        # Check regular info dict (PNG tEXt chunks, etc.)
        for k, v in img.info.items():
            try:
                metadata[str(k)] = str(v)[:500]
            except Exception:
                metadata[str(k)] = '<unreadable>'

        findings = []
        score = 50
        has_signal = False

        # Photo editors
        editors = ['photoshop', 'gimp', 'lightroom', 'affinity', 'canva',
                   'snapseed', 'capture one', 'darktable', 'rawtherapee']

        # Camera/phone brands (mínimo de 4 caracteres para evitar colisão de substring,
        # ex: 'lg' casando dentro de 'light'/'background')
        camera_brands = ['canon', 'nikon', 'sony', 'fujifilm', 'olympus', 'panasonic',
                         'leica', 'pentax', 'hasselblad', 'apple', 'samsung', 'google',
                         'xiaomi', 'huawei', 'oppo', 'oneplus', 'motorola', 'realme', 'vivo']

        if not metadata:
            score = 50
            has_signal = False
            findings.append("Nenhum metadado presente (comum em mídias compartilhadas na web/redes sociais)")
        else:
            # meta_str_full: usado apenas para achar termos INEQUÍVOCOS (nomes compostos,
            # parâmetros com pontuação) que não geram falso positivo em texto livre.
            meta_str_full = str(metadata).lower()
            # meta_str_technical: exclui campos de texto livre (descrições/legendas/autoria)
            # e é o único lugar onde termos AMBÍGUOS (palavras comuns) são buscados.
            meta_str_technical = _build_scan_string(metadata)

            # 1. Check for explicit AI generator signatures
            found_ai = False
            for tag in UNAMBIGUOUS_AI_TAGS:
                if tag in meta_str_full:
                    score = 98
                    has_signal = True
                    findings.append(f"Assinatura de gerador AI detectada nos metadados: '{tag}'")
                    found_ai = True
                    break

            if not found_ai:
                for tag in AMBIGUOUS_AI_TAGS:
                    if _word_match(tag, meta_str_technical):
                        score = 90
                        has_signal = True
                        findings.append(
                            f"Termo associado a gerador de IA encontrado em campo técnico: '{tag}'"
                        )
                        found_ai = True
                        break

            # 2. Check for AI prompt/workflow structures in PNG text chunks
            if not found_ai:
                prompt_keys = ['parameters', 'prompt', 'workflow', 'comment', 'description']
                for key in prompt_keys:
                    if key in metadata:
                        val = str(metadata[key]).lower()
                        workflow_indicators = ['sampler', 'cfg', 'steps', 'seed', 'negative',
                                               'checkpoint', 'lora', 'scheduler', 'denoise']
                        matches = sum(1 for ind in workflow_indicators if ind in val)
                        if matches >= 3:
                            score = 95
                            has_signal = True
                            findings.append(f"Parâmetros de geração de IA detectados no campo '{key}'")
                            found_ai = True
                            break

            # 3. Check C2PA / Content Credentials & IPTC AI markers
            if not found_ai:
                xmp_ai_markers = ['digitalsourcetype', 'trainedalgorithmicmedia',
                                  'compositewithtrained', 'algorithmicmedia',
                                  'iptc:digitalsourcetype']
                for marker in xmp_ai_markers:
                    if marker in meta_str_full:
                        score = 92
                        has_signal = True
                        findings.append(f"Marcador IPTC/XMP de mídia algorítmica: '{marker}'")
                        found_ai = True
                        break

            # 4. Check Camera EXIF if no AI markers found
            if not found_ai:
                has_camera_make = any(k in metadata for k in ['Make', 'make'])
                has_camera_model = any(k in metadata for k in ['Model', 'model'])
                has_datetime = any(k in metadata for k in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized'])
                has_exposure = any(k in metadata for k in ['ExposureTime', 'FNumber', 'ISOSpeedRatings',
                                                           'FocalLength', 'ExposureProgram', 'Software'])
                has_gps = any('gps' in str(k).lower() for k in metadata.keys())

                if has_camera_make and has_camera_model and (has_exposure or has_datetime):
                    score = 8
                    has_signal = True
                    findings.append("Metadados EXIF autênticos de câmera/sensor fotográfico detectados")

                    # Identify specific brand (busca em campos técnicos, por palavra inteira)
                    for brand in camera_brands:
                        if _word_match(brand, meta_str_technical):
                            score = 5
                            findings.append(f"Dispositivo de captura identificado: {brand.capitalize()}")
                            break

                    if has_gps:
                        score = max(score - 3, 2)
                        findings.append("Geolocalização GPS original presente (forte indício de captura física)")

                elif has_camera_make or has_camera_model:
                    score = 22
                    has_signal = True
                    findings.append("Metadados parciais de equipamento fotográfico encontrados")

                else:
                    # Generic metadata (e.g. dimensions, color profile) without camera or AI info
                    score = 50
                    has_signal = False
                    findings.append("Metadados técnicos padrão sem registros de câmera ou IA (neutro)")

                # Note photo editors if present (busca em campos técnicos, ex: tag Software)
                for editor in editors:
                    if editor in meta_str_technical:
                        findings.append(f"Software de edição identificado: {editor.capitalize()}")
                        break

        return {
            'score': min(max(int(score), 0), 100),
            'has_signal': has_signal,
            'findings': findings if findings else ['Análise de metadados concluída'],
            'details': {'metadata': metadata}
        }

    except Exception as e:
        return {
            'score': 50,
            'has_signal': False,
            'findings': [f"Erro na leitura de metadados: {str(e)}"],
            'details': {'metadata': {}}
        }