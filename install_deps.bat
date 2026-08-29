@echo off
echo ============================================================
echo  Spectra — Instalando dependencias para wavelet + CLIP
echo ============================================================
echo.

echo [1/2] Instalando open-clip-torch (inclui PyTorch CPU)...
echo Aviso: primeiro download ~500MB (PyTorch + modelo CLIP)
echo        Armazenado em cache apos a primeira execucao.
echo.
python -m pip install open-clip-torch

echo.
echo [2/2] Verificando instalacao...
python -c "import open_clip; print('[OK] open_clip versao:', open_clip.__version__)"
python -c "import torch; print('[OK] torch versao:', torch.__version__)"

echo.
echo ============================================================
echo  Instalacao concluida!
echo.
echo  Proximos passos:
echo    1. python build_clip_prototypes.py   (gerar prototipos do dataset)
echo    2. python app.py                     (iniciar servidor)
echo ============================================================
pause
