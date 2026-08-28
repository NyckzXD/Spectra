from app import app

if __name__ == '__main__':
    print("=" * 50)
    print("  SPECTRA — AI Image Forensic Analyzer")
    print("  Server running at http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
