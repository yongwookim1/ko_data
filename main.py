#!/usr/bin/env python3
"""
Korean Image Data Crawler - Main Entry Point

Usage:
    python main.py [crawler] [--options]
    
Crawlers:
    dc      - DC Inside crawler
    ilbe    - Ilbe crawler  
    namu    - NamuWiki crawler
    google  - Google Image crawler (based on NamuWiki controversy titles)
    filter  - Run image filtering with VLM
    merge   - Merge filtered images
    all     - Run all crawlers sequentially
"""

import sys
import argparse
import logging

logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def run_dc():
    from crawlers import DCCrawler
    DCCrawler().run()


def run_ilbe():
    from crawlers import IlbeCrawler
    IlbeCrawler().run()


def run_namu():
    from crawlers import NamuCrawler
    NamuCrawler().run()


def run_google():
    from crawlers import GoogleImageCrawler
    GoogleImageCrawler().run()


def run_filter():
    from processing import ImageFilter
    ImageFilter().run()


def run_merge():
    from processing import merge_images
    from config import IMAGE_DIR
    sources = [IMAGE_DIR / "filtered" / "safe", IMAGE_DIR / "filtered" / "unsafe_usable"]
    merge_images(sources)


def main():
    parser = argparse.ArgumentParser(description="Korean Image Data Crawler")
    parser.add_argument('crawler', nargs='?', default='help',
                        choices=['dc', 'ilbe', 'namu', 'google', 'filter', 'merge', 'all', 'help'],
                        help='Crawler to run')
    args = parser.parse_args()

    runners = {
        'dc': run_dc,
        'ilbe': run_ilbe,
        'namu': run_namu,
        'google': run_google,
        'filter': run_filter,
        'merge': run_merge,
    }

    if args.crawler == 'help':
        parser.print_help()
        return

    if args.crawler == 'all':
        for name, runner in runners.items():
            if name not in ['filter', 'merge']:
                logging.info(f"=== Starting {name.upper()} crawler ===")
                try:
                    runner()
                except Exception as e:
                    logging.error(f"{name} failed: {e}")
    else:
        runners[args.crawler]()


if __name__ == "__main__":
    main()

