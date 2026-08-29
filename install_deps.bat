@echo off
echo ============================================================
echo  Spectra — Instalando dependencias (PyTorch, Transformers, CLIP)
echo ============================================================
echo.

echo [1/2] Instalando pacotes...
python -m pip install -r requirements.txt

echo.
echo [2/2] Verificando instalacao...
python -c "import torch; print('[OK] torch:', torch.__version__)"
python -c "import transformers; print('[OK] transformers:', transformers.__version__)"

echo.
echo ============================================================
echo  Instalacao concluida com sucesso!
echo.
echo  Inicie o servidor diretamente com:
echo    python app.py
echo ============================================================
pause
