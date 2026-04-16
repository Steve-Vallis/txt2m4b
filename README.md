# txt2m4b

Convert `.txt`, `.pdf`, and `.epub` files into `.m4b` audiobooks using [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) for text-to-speech and `ffmpeg` for final audiobook packaging.

The script:

- extracts text from plain text, PDF, and EPUB files
- applies basic cleanup for OCR and formatting issues
- generates speech with Kokoro TTS
- builds an `.m4b` audiobook with title metadata and optional cover art

## Features

- Supports `.txt`, `.pdf`, and `.epub` input files
- Uses Kokoro ONNX voices locally
- Runs well on Apple Silicon Macs and is designed for efficient local conversion
- Attempts to extract title, author, and cover art from source files
- Falls back to Open Library cover search when needed
- Outputs standard AAC-based `.m4b` files
- Lets you control voice, speech speed, bitrate, and chunk size

## Requirements

- Python 3.10+
- `ffmpeg` installed and available on your `PATH`

Python dependencies are listed in [`requirements.txt`](requirements.txt).

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install Python dependencies.
4. Make sure `ffmpeg` is installed.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS with Homebrew:

```bash
brew install ffmpeg
```

## Usage

Basic usage:

```bash
python3 txt2m4b.py /path/to/book.epub
```

This will create an `.m4b` file named after the detected title, or the input filename if no title metadata is found.

### Examples

Convert a text file with the default voice:

```bash
python3 txt2m4b.py manuscript.txt
```

Set a different voice and faster speech rate:

```bash
python3 txt2m4b.py novel.pdf --voice af_bella --speed 1.15
```

Write to a custom output file and bitrate:

```bash
python3 txt2m4b.py book.epub --output my-audiobook.m4b --bitrate 96k
```

Skip cover extraction and cover lookup:

```bash
python3 txt2m4b.py book.pdf --skip-cover
```

Adjust chunk size for longer or shorter TTS segments:

```bash
python3 txt2m4b.py book.txt --max-chars 600
```

## CLI Options

```text
positional arguments:
  input                 Input file (.txt, .pdf, .epub)

options:
  -v, --voice           Kokoro voice to use (default: af_sarah)
  -s, --speed           TTS speed (default: 1.0)
  -o, --output          Output M4B file path
  -b, --bitrate         Audio bitrate for M4B (default: 64k)
  --skip-cover          Skip fetching or extracting cover art
  --max-chars           Max characters per TTS chunk (default: 800)
```

## How It Works

1. The script reads text from the input file.
2. It normalizes formatting and applies basic cleanup.
3. Long paragraphs are split into smaller chunks for TTS safety.
4. Kokoro generates WAV audio for each chunk.
5. `ffmpeg` packages the audio into an `.m4b` file with metadata and optional cover art.

## Model Files

The script looks for these files in the project directory:

- `kokoro-v1.0.onnx`
- `voices-v1.0.bin`

If they are missing, it downloads them automatically from the Kokoro ONNX release assets.

## Notes

- Apple Silicon: this project runs locally and is a good fit for Apple M-series Macs. The current script does not explicitly configure Apple-specific acceleration backends, so performance depends on your Python and ONNX Runtime setup.
- PDF extraction quality depends heavily on the source document. Scanned PDFs without embedded text will not work unless OCR has already been applied.
- Cover handling is best-effort. The script first tries embedded or generated cover art, then falls back to Open Library lookup.
- The script currently generates English speech with `lang="en-us"`.

## Project Files

- [`txt2m4b.py`](txt2m4b.py): main conversion script
- [`requirements.txt`](requirements.txt): Python dependencies
- `kokoro-v1.0.onnx`: Kokoro ONNX model
- `voices-v1.0.bin`: Kokoro voice data

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
