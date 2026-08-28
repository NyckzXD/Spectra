import traceback
from PIL import Image
from PIL.ExifTags import TAGS


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

        # --- Comprehensive AI generator tags ---
        ai_tags = [
            # Diffusion-based & engines
            'stable diffusion', 'dall-e', 'dall·e', 'midjourney', 'comfyui', 'automatic1111',
            'a1111', 'novelai', 'adobe firefly', 'flux.1', 'flux', 'invoke ai', 'invokeai',
            'foocus', 'fooocus', 'dreamstudio', 'dream studio',
            # Commercial / Web / API
            'craiyon', 'leonardo.ai', 'leonardo ai', 'playground ai', 'ideogram',
            'copilot designer', 'bing image creator', 'nightcafe', 'artbreeder',
            'runway', 'pika', 'sora', 'kling', 'recraft', 'grok', 'xai',
            'gemini', 'imagen', 'parti', 'muse',
            # Common workflow parameters
            'steps:', 'sampler:', 'cfg scale:', 'seed:', 'negative prompt:',
            'lora:', 'embedding:', 'vae:', 'clip skip'
        ]

        # Photo editors
        editors = ['photoshop', 'gimp', 'lightroom', 'affinity', 'canva',
                   'snapseed', 'capture one', 'darktable', 'rawtherapee']

        # Camera/phone brands
        camera_brands = ['canon', 'nikon', 'sony', 'fujifilm', 'olympus', 'panasonic',
                         'leica', 'pentax', 'hasselblad', 'apple', 'samsung', 'google',
                         'xiaomi', 'huawei', 'oppo', 'oneplus', 'motorola', 'lg', 'realme', 'vivo']

        if not metadata:
            score = 50
            has_signal = False
            findings.append("Nenhum metadado presente (comum em mídias compartilhadas na web/redes sociais)")
        else:
            meta_str = str(metadata).lower()

            # 1. Check for explicit AI generator signatures
            found_ai = False
            for tag in ai_tags:
                if tag in meta_str:
                    score = 98
                    has_signal = True
                    findings.append(f"Assinatura de gerador AI detectada nos metadados: '{tag}'")
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
                    if marker in meta_str:
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

                    # Identify specific brand
                    for brand in camera_brands:
                        if brand in meta_str:
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

                # Note photo editors if present
                for editor in editors:
                    if editor in meta_str:
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
