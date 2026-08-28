import traceback
from PIL import Image
from PIL.ExifTags import TAGS


def analyze_metadata(image_path: str) -> dict:
    """
    Analyzes image metadata to detect AI generation signatures.
    Checks EXIF, PNG tEXt chunks, IPTC/XMP markers, C2PA, and known AI generator tags.
    Returns: {'score': int, 'findings': [str], 'details': {'metadata': {key: value}}}
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
        score = 0

        # --- Comprehensive AI generator tags ---
        ai_tags = [
            # Diffusion-based
            'stable diffusion', 'dall-e', 'dall·e', 'midjourney', 'comfyui', 'automatic1111',
            'a1111', 'novelai', 'adobe firefly', 'flux', 'invoke ai', 'invokeai',
            'foocus', 'fooocus', 'dreamstudio', 'dream studio',
            # Commercial / API
            'craiyon', 'leonardo.ai', 'leonardo ai', 'playground ai', 'ideogram',
            'copilot designer', 'bing image creator', 'nightcafe', 'artbreeder',
            'runway', 'pika', 'sora', 'kling', 'recraft', 'grok', 'xai',
            'gemini', 'imagen', 'parti', 'muse',
            # Video / multi-modal (sometimes in stills)
            'gen-2', 'gen-3', 'luma', 'haiper',
            # Common workflow markers
            'steps:', 'sampler:', 'cfg scale:', 'seed:', 'model:', 'negative prompt:',
            'lora:', 'embedding:', 'vae:', 'clip skip',
        ]

        # Photo editors (less suspicious than generators)
        editors = ['photoshop', 'gimp', 'lightroom', 'affinity', 'canva',
                   'snapseed', 'capture one', 'darktable', 'rawtherapee']

        # Camera/phone markers (strong indicator of real photo)
        camera_brands = ['canon', 'nikon', 'sony', 'fujifilm', 'olympus', 'panasonic',
                         'leica', 'pentax', 'hasselblad', 'apple', 'samsung', 'google',
                         'xiaomi', 'huawei', 'oppo', 'oneplus', 'motorola', 'lg']

        if not metadata:
            score = 65
            findings.append("Nenhum metadado encontrado — comum em screenshots e imagens AI")
        else:
            meta_str = str(metadata).lower()

            # --- Check for AI generator signatures ---
            found_ai = False
            for tag in ai_tags:
                if tag in meta_str:
                    score = 95
                    findings.append(f"Assinatura de gerador AI encontrada: {tag}")
                    found_ai = True
                    break

            if not found_ai:
                # --- PNG tEXt chunk analysis (ComfyUI, A1111 embed prompts here) ---
                prompt_keys = ['parameters', 'prompt', 'workflow', 'comment', 'description']
                for key in prompt_keys:
                    if key in metadata:
                        val = metadata[key].lower()
                        # Check if it contains workflow/prompt-like content
                        workflow_indicators = ['sampler', 'cfg', 'steps', 'seed', 'negative',
                                               'checkpoint', 'lora', 'scheduler', 'denoise']
                        matches = sum(1 for ind in workflow_indicators if ind in val)
                        if matches >= 3:
                            score = 92
                            findings.append(f"Workflow/prompt de gerador AI detectado no campo '{key}'")
                            found_ai = True
                            break

            if not found_ai:
                # --- Camera EXIF analysis ---
                has_camera_make = 'Make' in metadata or 'make' in metadata
                has_camera_model = 'Model' in metadata or 'model' in metadata
                has_datetime = 'DateTime' in metadata or 'DateTimeOriginal' in metadata
                has_exposure = any(k in metadata for k in
                                  ['ExposureTime', 'FNumber', 'ISOSpeedRatings',
                                   'FocalLength', 'ExposureProgram'])
                has_gps = any('gps' in k.lower() for k in metadata.keys())

                # Rich EXIF = very likely real
                if has_camera_make and has_camera_model and has_exposure:
                    score = 8
                    findings.append("EXIF completo de câmera encontrado (Make, Model, Exposure)")

                    # Check for known camera brands
                    for brand in camera_brands:
                        if brand in meta_str:
                            score = 5
                            findings.append(f"Marca de câmera/dispositivo identificada: {brand}")
                            break

                    if has_gps:
                        score = max(score - 5, 2)
                        findings.append("Dados GPS encontrados — forte indicador de foto real")

                elif has_camera_make or has_camera_model:
                    score = 15
                    findings.append("EXIF parcial de câmera encontrado")
                elif not has_camera_make and not has_camera_model and not has_datetime:
                    score = 70
                    findings.append("Campos EXIF de câmera e data ausentes")
                else:
                    score = 50
                    findings.append("Metadados inconclusivos — sem informações de câmera claras")

                # --- Editor detection ---
                for editor in editors:
                    if editor in meta_str:
                        # Editor doesn't mean AI, but raises slight suspicion
                        score = max(score, 40)
                        findings.append(f"Editor de imagem detectado: {editor}")
                        break

            # --- C2PA / Content Credentials ---
            c2pa_markers = ['c2pa', 'contentcredentials', 'content credentials',
                            'content authenticity', 'cai', 'cr:']
            for marker in c2pa_markers:
                if marker in meta_str:
                    findings.append("Marcadores C2PA / Content Credentials encontrados")
                    break

            # --- IPTC/XMP AI markers ---
            xmp_ai_markers = ['digitalsourcetype', 'trainedalgorithmicmedia',
                              'compositewithtrained', 'algorithmicmedia',
                              'iptc:digitalsourcetype']
            for marker in xmp_ai_markers:
                if marker in meta_str:
                    score = max(score, 85)
                    findings.append(f"Marcador IPTC/XMP de mídia algorítmica: {marker}")
                    break

        return {
            'score': min(max(int(score), 0), 100),
            'findings': findings if findings else ['Análise de metadados concluída sem achados significativos'],
            'details': {'metadata': metadata}
        }

    except Exception as e:
        return {
            'score': 50,
            'findings': [f"Erro na análise de metadados: {str(e)}"],
            'details': {'metadata': {}}
        }
