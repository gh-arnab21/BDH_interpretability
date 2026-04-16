#!/usr/bin/env python3
"""
Europarl Dataset Downloader - FIXED VERSION

Downloads and preprocesses the Europarl parallel corpus for BDH training.
Supports English-French (en-fr) and English-Portuguese (en-pt) language pairs.

Usage:
    python download_europarl.py --languages en-fr en-pt --output ./data
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple
import requests
from tqdm import tqdm
import tarfile
import gzip


# Europarl download URLs (from statmt.org)
EUROPARL_URLS = {
    "en-fr": {
        "url": "https://www.statmt.org/europarl/v7/fr-en.tgz",
        "src_pattern": ".en",
        "tgt_pattern": ".fr",
    },
    "en-pt": {
        "url": "https://www.statmt.org/europarl/v7/pt-en.tgz",
        "src_pattern": ".en",
        "tgt_pattern": ".pt",
    },
    "en-es": {
        "url": "https://www.statmt.org/europarl/v7/es-en.tgz",
        "src_pattern": ".en",
        "tgt_pattern": ".es",
    },
    "en-de": {
        "url": "https://www.statmt.org/europarl/v7/de-en.tgz",
        "src_pattern": ".en",
        "tgt_pattern": ".de",
    },
}


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_europarl_direct(lang_pair: str, output_dir: Path) -> Tuple[Path, Path]:
    """
    Download Europarl directly from statmt.org.
    """
    if lang_pair not in EUROPARL_URLS:
        raise ValueError(f"Unsupported language pair: {lang_pair}")
    
    config = EUROPARL_URLS[lang_pair]
    url = config["url"]
    
    pair_dir = output_dir / lang_pair
    pair_dir.mkdir(parents=True, exist_ok=True)
    
    archive_path = pair_dir / "europarl.tgz"
    
    # Download if not exists
    if not archive_path.exists():
        print(f"\n📥 Downloading {lang_pair} from statmt.org...")
        if not download_file(url, archive_path, f"Downloading {lang_pair}"):
            raise RuntimeError(f"Failed to download {url}")
    else:
        print(f"📦 Using cached archive: {archive_path}")
    
    # Extract
    print(f"📦 Extracting archive...")
    with tarfile.open(archive_path, 'r:gz') as tar:
        # List contents first
        members = tar.getmembers()
        print(f"   Archive contains {len(members)} files")
        
        # Extract all files with filter for Python 3.14 compatibility
        for member in tqdm(members, desc="   Extracting"):
            try:
                tar.extract(member, path=pair_dir, filter='data')
            except TypeError:
                # Older Python without filter argument
                tar.extract(member, path=pair_dir)
    
    # Find the extracted files
    src_pattern = config["src_pattern"]
    tgt_pattern = config["tgt_pattern"]
    
    src_file = None
    tgt_file = None
    
    # Search recursively for the language files
    print(f"🔍 Searching for extracted files...")
    for f in pair_dir.rglob("*"):
        if f.is_file():
            fname = f.name.lower()
            if fname.endswith(src_pattern):
                src_file = f
                print(f"   Found source: {f}")
            elif fname.endswith(tgt_pattern):
                tgt_file = f
                print(f"   Found target: {f}")
    
    if not src_file or not tgt_file:
        # List what we actually have
        print("\n   Files in directory:")
        for f in pair_dir.rglob("*"):
            if f.is_file():
                print(f"     {f}")
        raise RuntimeError(f"Could not find extracted files in {pair_dir}")
    
    return src_file, tgt_file


def prepare_bdh_format(
    src_file: Path,
    tgt_file: Path,
    output_file: Path,
    src_lang: str,
    tgt_lang: str,
    max_pairs: int = None
) -> int:
    """
    Convert parallel corpus to BDH training format.
    
    BDH format:
    <F:en>English sentence.<T:fr>French translation.<F:en>Next sentence...
    """
    print(f"\n🔄 Converting to BDH format...")
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    src_lines = None
    tgt_lines = None
    
    for encoding in encodings:
        try:
            with open(src_file, 'r', encoding=encoding) as f:
                src_lines = f.readlines()
            with open(tgt_file, 'r', encoding=encoding) as f:
                tgt_lines = f.readlines()
            print(f"   Using encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if src_lines is None or tgt_lines is None:
        raise RuntimeError("Could not read files with any encoding")
    
    print(f"   Source lines: {len(src_lines):,}")
    print(f"   Target lines: {len(tgt_lines):,}")
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        count = 0
        for src_line, tgt_line in tqdm(zip(src_lines, tgt_lines), 
                                        total=min(len(src_lines), len(tgt_lines)),
                                        desc="   Converting"):
            src_text = src_line.strip()
            tgt_text = tgt_line.strip()
            
            if not src_text or not tgt_text:
                continue
            
            # Skip very long lines (likely parsing errors)
            if len(src_text) > 1000 or len(tgt_text) > 1000:
                continue
            
            # BDH format with language markers
            formatted = f"<F:{src_lang}>{src_text}<T:{tgt_lang}>{tgt_text}"
            out_f.write(formatted)
            
            count += 1
            if max_pairs and count >= max_pairs:
                break
    
    return count


def create_train_val_split(
    input_file: Path,
    output_dir: Path,
    val_ratio: float = 0.1
) -> Tuple[Path, Path]:
    """Split data into training and validation sets."""
    
    print(f"\n✂️ Creating train/val split (val_ratio={val_ratio})...")
    
    with open(input_file, 'rb') as f:
        data = f.read()
    
    split_point = int(len(data) * (1 - val_ratio))
    
    train_file = output_dir / "train.bin"
    val_file = output_dir / "val.bin"
    
    with open(train_file, 'wb') as f:
        f.write(data[:split_point])
    
    with open(val_file, 'wb') as f:
        f.write(data[split_point:])
    
    print(f"   Train: {len(data[:split_point]):,} bytes ({train_file})")
    print(f"   Val: {len(data[split_point:]):,} bytes ({val_file})")
    
    return train_file, val_file


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare Europarl dataset for BDH training"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en-fr"],
        help="Language pairs to download (e.g., en-fr en-pt)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data"),
        help="Output directory"
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum sentence pairs per language (for testing)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🐉 BDH Europarl Dataset Downloader (Fixed)")
    print("=" * 60)
    print(f"Languages: {args.languages}")
    print(f"Output: {args.output}")
    print(f"Max pairs: {args.max_pairs or 'unlimited'}")
    print("=" * 60)
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    for lang_pair in args.languages:
        print(f"\n{'=' * 60}")
        print(f"Processing {lang_pair.upper()}")
        print("=" * 60)
        
        src_lang, tgt_lang = lang_pair.split("-")
        pair_dir = args.output / lang_pair
        pair_dir.mkdir(parents=True, exist_ok=True)
        
        # Download
        try:
            src_file, tgt_file = download_europarl_direct(lang_pair, args.output)
        except Exception as e:
            print(f"❌ Failed to download {lang_pair}: {e}")
            continue
        
        # Convert to BDH format
        bdh_file = pair_dir / "europarl_bdh.txt"
        num_pairs = prepare_bdh_format(
            src_file, tgt_file, bdh_file,
            src_lang, tgt_lang,
            args.max_pairs
        )
        print(f"✅ Converted {num_pairs:,} sentence pairs")
        
        # Create train/val split
        train_file, val_file = create_train_val_split(
            bdh_file, pair_dir, args.val_ratio
        )
        
        # Print statistics
        print(f"\n📊 Statistics for {lang_pair}:")
        print(f"   Total pairs: {num_pairs:,}")
        print(f"   Train file: {train_file}")
        print(f"   Val file: {val_file}")
    
    print("\n" + "=" * 60)
    print("✅ Download complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Train French model:    python train.py --config configs/french.yaml")
    print("  2. Train Portuguese model: python train.py --config configs/portuguese.yaml")


if __name__ == "__main__":
    main()
