import os
import sys
import subprocess
import requests
import soundfile as sf
import numpy as np
from pathlib import Path
from kokoro_onnx import Kokoro
import onnxruntime as ort
import re
import fitz  # PyMuPDF
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import warnings
import argparse

# Configuration
VOICE = "af_sarah"
SPEED = 1.0
GITHUB_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
GITHUB_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

# Silence ebooklib warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {dest}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF and attempt some basic cleanup."""
    print(f"Extracting text from PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    full_text = []
    
    # Try to extract metadata
    metadata = doc.metadata
    title = metadata.get("title", "")
    author = metadata.get("author", "")
    
    # Simple header/footer detection: 
    # Compare common lines at top and bottom across first 5 pages
    potential_headers = []
    potential_footers = []
    
    for i in range(min(5, len(doc))):
        page = doc.load_page(i)
        lines = page.get_text("text").split("\n")
        if lines:
            potential_headers.append(lines[0].strip())
            potential_footers.append(lines[-1].strip())
            
    header_to_skip = None
    footer_to_skip = None
    
    if len(potential_headers) >= 3:
        # Check if same header appears at least 3 times
        from collections import Counter
        common_h = Counter(potential_headers).most_common(1)
        if common_h and common_h[0][1] >= 3:
            header_to_skip = common_h[0][0]
            
        common_f = Counter(potential_footers).most_common(1)
        if common_f and common_f[0][1] >= 3:
            footer_to_skip = common_f[0][0]

    for page in doc:
        text = page.get_text("text")
        lines = text.split("\n")
        
        # Filter out obvious page numbers and headers/footers
        cleaned_lines = []
        for line in lines:
            l = line.strip()
            if not l: continue
            if l == header_to_skip or l == footer_to_skip: continue
            if re.match(r'^\d+$', l): continue # Only digits (page number)
            if re.match(r'^Page \d+$', l, re.IGNORECASE): continue
            cleaned_lines.append(l)
            
        full_text.append("\n".join(cleaned_lines))
        
    # Attempt to get cover image from first page
    cover_file = None
    page = doc.load_page(0)
    pix = page.get_pixmap()
    cover_file = "cover.jpg"
    pix.save(cover_file)

    return "\n\n".join(full_text), title, author, cover_file

def extract_text_from_epub(epub_path):
    """Extract text from EPUB and attempt some basic cleanup."""
    print(f"Extracting text from EPUB: {epub_path}")
    book = epub.read_epub(epub_path)
    title = book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else ""
    author = book.get_metadata('DC', 'creator')[0][0] if book.get_metadata('DC', 'creator') else ""
    
    full_text = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text from paragraphs, headings, etc.
        # This keeps some structure
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for p in paragraphs:
            text = p.get_text().strip()
            if text:
                full_text.append(text)
                
    # Try to find cover image
    cover_file = None
    # ebooklib doesn't make this trivial, but we can look for items with 'cover' in name
    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        if 'cover' in item.get_name().lower():
            cover_file = "cover.jpg"
            with open(cover_file, "wb") as f:
                f.write(item.get_content())
            break

    return "\n\n".join(full_text), title, author, cover_file

def get_cover_art(title, author=""):
    # If we already have a cover.jpg from extraction, use it
    if os.path.exists("cover.jpg"):
        # Check if it's large enough or if we should try to find a better one
        # For now, let's assume if it exists, it's good
        return "cover.jpg"

    print(f"Searching cover art for: {title} by {author}")
    query = f"{title} {author}".strip()
    search_url = f"https://openlibrary.org/search.json?q={requests.utils.quote(query)}"
    try:
        response = requests.get(search_url).json()
        if response.get('docs'):
            book = response['docs'][0]
            cover_id = book.get('cover_i')
            if cover_id:
                cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
                print(f"Found cover art: {cover_url}")
                img_data = requests.get(cover_url).content
                with open("cover.jpg", "wb") as f:
                    f.write(img_data)
                return "cover.jpg"
            
            isbn = book.get('isbn')
            if isbn:
                cover_url = f"https://covers.openlibrary.org/b/isbn/{isbn[0]}-L.jpg"
                print(f"Found cover art via ISBN: {cover_url}")
                img_data = requests.get(cover_url).content
                with open("cover.jpg", "wb") as f:
                    f.write(img_data)
                return "cover.jpg"
    except Exception as e:
        print(f"Error fetching cover art: {e}")
    
    print("No cover art found.")
    return None

def preprocess_text(text):
    """Initial pass to clean up global text issues."""
    # 1. Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 2. Join hyphenated words at line breaks
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # 3. Replace ligatures
    ligatures = {
        'ﬀ': 'ff', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
        'ﬅ': 'ft', 'ﬆ': 'st',
    }
    for ligature, replacement in ligatures.items():
        text = text.replace(ligature, replacement)

    # 4. Replace smart quotes and dashes
    replacements = {
        '“': '"', '”': '"', '‘': "'", '’': "'",
        '—': ' - ', '–': ' - ', '…': '...',
        '\u00a0': ' ',  # non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
        
    return text

def clean_text(text):
    """Clean up individual paragraphs."""
    # Remove footnote markers like wordiv or 2024.v
    # Matches lowercase roman numerals at the end of words or sentences
    text = re.sub(r'(\d+)([ivx]{1,4})\b', r'\1', text)
    text = re.sub(r'([a-zA-Z]{3,})\.([ivx]{1,4})\b', r'\1.', text)
    
    # Fix obvious OCR artifacts (broken words)
    # Target common ones seen in the text
    text = text.replace('e xcept', 'except')
    text = text.replace('acquir ed', 'acquired')
    text = text.replace('befor e', 'before')
    text = text.replace('giveawa ys', 'giveaways')
    text = text.replace('MITIGA TING', 'MITIGATING')
    text = text.replace('isits', 'is its')
    text = text.replace('F ootnote', 'Footnote')
    
    # Generic fix for single characters split from words (e.g., "masterin g" -> "mastering")
    # Only for characters that are likely suffixes or parts of words
    text = re.sub(r'(\w{3,}) ([bcdefghjklmnopqrstuvwxyz])\b', r'\1\2', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def split_into_paragraphs(text):
    """Split text into coherent paragraphs."""
    # Split by double newline (or more) to identify true paragraphs
    raw_paragraphs = re.split(r'\n\s*\n', text)
    
    paragraphs = []
    for p in raw_paragraphs:
        # Join single newlines within a paragraph
        p = p.replace('\n', ' ').strip()
        if p:
            paragraphs.append(p)
    return paragraphs

def chunk_text(text, max_chars=800):
    """Split a paragraph into chunks that are safe for TTS, avoiding middle-of-sentence splits."""
    if len(text) <= max_chars:
        return [text]
    
    # Split by sentence endings (. ! ?) followed by space
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            # If a single sentence is still too long, hard split it
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i:i+max_chars])
                current_chunk = ""
            else:
                current_chunk = sentence
                
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Convert text, PDF, or EPUB to M4B audiobook using Kokoro TTS.")
    parser.add_argument("input", type=Path, help="Input file (.txt, .pdf, .epub)")
    parser.add_argument("-v", "--voice", default="af_sarah", help="Kokoro voice to use (default: af_sarah)")
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="TTS speed (default: 1.0)")
    parser.add_argument("-o", "--output", type=Path, help="Output M4B file path (default: <input_title>.m4b)")
    parser.add_argument("-b", "--bitrate", default="64k", help="Audio bitrate for M4B (default: 64k)")
    parser.add_argument("--skip-cover", action="store_true", help="Skip fetching or extracting cover art")
    parser.add_argument("--max-chars", type=int, default=800, help="Max characters per TTS chunk (default: 800)")
    
    args = parser.parse_args()

    input_file = args.input
    if not input_file.exists():
        print(f"File not found: {input_file}")
        sys.exit(1)

    voice = args.voice
    speed = args.speed
    bitrate = args.bitrate
    max_chars = args.max_chars

    title = input_file.stem
    author = ""
    cover_file = None
    full_text = ""
    
    # Clean up any old cover file
    if os.path.exists("cover.jpg"):
        os.remove("cover.jpg")

    # Load content based on file type
    suffix = input_file.suffix.lower()
    if suffix == '.txt':
        print(f"Reading text file: {input_file}...")
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            full_text = f.read()
        # Try to find author in first few lines
        lines = full_text.split("\n")[:20]
        for line in lines:
            if "by " in line.lower():
                author = line.lower().split("by ")[1].strip()
                break
    elif suffix == '.pdf':
        full_text, pdf_title, pdf_author, cover_file = extract_text_from_pdf(input_file)
        if pdf_title: title = pdf_title
        if pdf_author: author = pdf_author
    elif suffix == '.epub':
        full_text, epub_title, epub_author, cover_file = extract_text_from_epub(input_file)
        if epub_title: title = epub_title
        if epub_author: author = epub_author
    else:
        print(f"Unsupported file format: {suffix}")
        sys.exit(1)

    if not full_text.strip():
        print("No text content found in the file.")
        sys.exit(1)

    # Setup models
    model_path = "kokoro-v1.0.onnx"
    voices_path = "voices-v1.0.bin"
    download_file(GITHUB_MODEL_URL, model_path)
    download_file(GITHUB_VOICES_URL, voices_path)

    # Initialize Kokoro
    print("Initializing Kokoro TTS...")
    try:
        kokoro = Kokoro(model_path, voices_path)
    except Exception as e:
        print(f"Error initializing Kokoro: {e}")
        sys.exit(1)

    # Get cover art if not extracted and not skipped
    if not args.skip_cover:
        if not cover_file or not os.path.exists(cover_file):
            cover_file = get_cover_art(title, author)
    else:
        cover_file = None

    print("Pre-processing text...")
    full_text = preprocess_text(full_text)
    
    paragraphs = split_into_paragraphs(full_text)
    print(f"Processing {len(paragraphs)} paragraphs...")

    temp_wav = "temp_combined.wav"
    sample_rate = 24000 # Kokoro standard
    
    print("Generating audio (this may take a while)...")
    with sf.SoundFile(temp_wav, mode='w', samplerate=sample_rate, channels=1) as out_f:
        for i, p in enumerate(paragraphs):
            p = clean_text(p)
            if not p: continue
            
            try:
                chunks = chunk_text(p, max_chars=max_chars)
                for j, chunk in enumerate(chunks):
                    if not chunk.strip(): continue
                    
                    # Print progress
                    pct = (i/len(paragraphs)*100 + (j/len(chunks))*(100/len(paragraphs)))
                    if len(paragraphs) > 1:
                        msg = f"Paragraph {i+1}/{len(paragraphs)} | Chunk {j+1}/{len(chunks)} ({pct:.1f}%)"
                    else:
                        msg = f"Chunk {j+1}/{len(chunks)} ({pct:.1f}%)"
                    
                    print(msg + " " * (80 - len(msg)), end="\r", flush=True)
                        
                    samples, sr = kokoro.create(chunk, voice=voice, speed=speed, lang="en-us")
                    out_f.write(samples)
            except Exception as e:
                print(f"\nError processing paragraph {i}: {e}")
    
    print("\nAudio generation complete.")

    # Convert to M4B using ffmpeg
    output_m4b = args.output if args.output else f"{title}.m4b"
    print(f"Assembling M4B: {output_m4b}")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_wav,
    ]
    
    if cover_file and os.path.exists(cover_file):
        cmd.extend(["-i", cover_file])
    
    cmd.extend([
        "-map", "0:a",
    ])
    
    if cover_file and os.path.exists(cover_file):
        cmd.extend(["-map", "1:0", "-c:v", "copy", "-disposition:v:0", "attached_pic"])
        
    cmd.extend([
        "-c:a", "aac",
        "-b:a", bitrate,
        "-metadata", f"title={title}",
    ])
    
    if author:
        cmd.extend(["-metadata", f"artist={author}"])
    
    cmd.append(str(output_m4b))

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Cleanup
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

    print(f"Successfully created: {output_m4b}")

if __name__ == "__main__":
    main()
