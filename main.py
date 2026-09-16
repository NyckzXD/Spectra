from app import app, _run_config

if __name__ == '__main__':
    cfg = _run_config()
    print("=" * 50)
    print("  SPECTRA — AI Image Forensic Analyzer")
    print(f"  Server running at http://{cfg['host']}:{cfg['port']}")
    if cfg['debug']:
        print("  Modo debug ATIVO (apenas local)")
    print("=" * 50)
    app.run(**cfg)
