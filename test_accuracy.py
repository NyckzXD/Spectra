import os
import io
import numpy as np
from PIL import Image, PngImagePlugin
from app import calculate_composite_score, get_verdict, calculate_concordance
from analyzers.metadata_analyzer import analyze_metadata
from analyzers.noise_analyzer import analyze_noise
from analyzers.spectral_analyzer import analyze_spectral
from analyzers.statistical_analyzer import analyze_statistical
from analyzers.ela_analyzer import analyze_ela
from analyzers.artifact_analyzer import analyze_artifacts


def run_all_analyzers(path):
    analyses = {
        'metadata': analyze_metadata(path),
        'noise': analyze_noise(path),
        'spectral': analyze_spectral(path),
        'statistical': analyze_statistical(path),
        'ela': analyze_ela(path),
        'artifacts': analyze_artifacts(path),
    }
    score = calculate_composite_score(analyses)
    verdict, level = get_verdict(score)
    confidence, agreement = calculate_concordance(analyses)
    return score, verdict, level, confidence, analyses


def create_simulated_real_image(path, with_exif=True):
    """Simulate optical photo with Poisson noise, natural 1/f^2 spectrum, and camera EXIF."""
    w, h = 512, 512
    # Create smooth natural gradient and objects
    y, x = np.mgrid[0:h, 0:w]
    bg = 120 + 80 * np.sin(x / 60.0) * np.cos(y / 80.0)
    bg = np.clip(bg, 20, 230)

    # Add Poisson-Gaussian noise: noise variance increases with intensity
    shot_noise = np.random.normal(0, 1.0, (h, w)) * np.sqrt(bg / 10.0 + 1.0)
    r = np.clip(bg + shot_noise + np.random.normal(0, 1.5, (h, w)), 0, 255)
    g = np.clip(bg * 0.9 + shot_noise + np.random.normal(0, 1.5, (h, w)), 0, 255)
    b = np.clip(bg * 0.8 + shot_noise + np.random.normal(0, 1.5, (h, w)), 0, 255)

    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    img = Image.fromarray(rgb)

    if with_exif:
        exif = img.getexif()
        exif[271] = 'Canon'  # Make
        exif[272] = 'Canon EOS R5'  # Model
        exif[306] = '2025:05:12 14:30:00'
        exif[33434] = (1, 250)  # ExposureTime
        exif[33437] = (28, 10)  # FNumber
        exif[34855] = 400       # ISO
        img.save(path, 'JPEG', quality=92, exif=exif)
    else:
        img.save(path, 'JPEG', quality=90)


def create_simulated_ai_diffusion_image(path, with_metadata=True):
    """Simulate AI diffusion output with VAE chrominance smoothing, high saturation, and AI metadata."""
    w, h = 512, 512
    y, x = np.mgrid[0:h, 0:w]
    # Smooth synthetic pattern with high saturation
    r = np.clip(180 + 70 * np.sin(x / 30.0), 0, 255)
    g = np.clip(60 + 50 * np.cos(y / 30.0), 0, 255)
    b = np.clip(220 + 30 * np.sin((x + y) / 40.0), 0, 255)

    # Uniform independent noise (non-Poisson)
    r += np.random.normal(0, 0.4, (h, w))
    g += np.random.normal(0, 0.4, (h, w))
    b += np.random.normal(0, 0.4, (h, w))

    rgb = np.stack([np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)], axis=-1).astype(np.uint8)
    img = Image.fromarray(rgb)

    if with_metadata:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", "photorealistic portrait, masterpiece, steps: 30, sampler: DPM++ 2M Karras, cfg scale: 7.0, seed: 42918274")
        pnginfo.add_text("Software", "ComfyUI")
        img.save(path, 'PNG', pnginfo=pnginfo)
    else:
        img.save(path, 'PNG')


if __name__ == '__main__':
    print("=== SPECTRA CALIBRATION VALIDATION TEST ===")
    os.makedirs("test_samples", exist_ok=True)

    real_exif_path = "test_samples/sample_real_camera.jpg"
    real_no_exif_path = "test_samples/sample_real_web_no_exif.jpg"
    ai_meta_path = "test_samples/sample_ai_prompt.png"
    ai_no_meta_path = "test_samples/sample_ai_stripped.png"

    create_simulated_real_image(real_exif_path, with_exif=True)
    create_simulated_real_image(real_no_exif_path, with_exif=False)
    create_simulated_ai_diffusion_image(ai_meta_path, with_metadata=True)
    create_simulated_ai_diffusion_image(ai_no_meta_path, with_metadata=False)

    tests = [
        ("Foto Real (com EXIF Canon R5)", real_exif_path),
        ("Foto Real da Web (sem EXIF / limpo)", real_no_exif_path),
        ("Imagem IA (com parâmetros ComfyUI / Prompt)", ai_meta_path),
        ("Imagem IA (sem metadados / redes sociais)", ai_no_meta_path),
    ]

    for name, p in tests:
        score, verdict, level, conf, analyses = run_all_analyzers(p)
        print(f"\n[{name}]")
        print(f"  -> Score Final: {score}% | Veredito: {verdict} ({level}) | Confiança: {conf}")
        print(f"     Detalhes por Analisador:")
        for k in ['metadata', 'noise', 'spectral', 'statistical', 'ela', 'artifacts']:
            s = analyses[k]['score']
            print(f"       - {k.capitalize()}: {s}%")

    print("\n=== TESTES CONCLUÍDOS ===")
