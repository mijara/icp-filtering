import os
import logging
import sys
from urllib.parse import urlparse
from urls import traverse_sitemap
from crawl import download_pages
from create import filter_files
import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure the logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Define log filename and path
log_filename = datetime.now().strftime("script_%Y%m%d_%H%M%S.log")
log_filepath = os.path.join(log_dir, log_filename)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    handlers=[
        logging.FileHandler(log_filepath),  
        logging.StreamHandler(sys.stdout)  
    ]
)

# Redirect stdout and stderr to logging
class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

def is_top_level(url, domain):
    parsed_url = urlparse(url)
    path = parsed_url.path.strip("/")
    segments = path.split("/")
    return len(segments) == 1

def process_website(website_url):
    try:
        website_url = website_url.strip()
        if not website_url:
            return

        sitemap_url = website_url + '/sitemap.xml'
        try:
            urls = traverse_sitemap(sitemap_url)
        except Exception as e:
            logging.error(f"Failed to traverse sitemap {sitemap_url}: {e}")
            return

        if urls:
            domain = website_url.split(".")[-2]
            file_path = os.path.join('websites', f'{domain}_urls.txt')
            with open(file_path, 'w') as file:
                for url in urls:
                    file.write(url + "\n")
            logging.info(f"URLs from {sitemap_url} written to {file_path}")
        else:
            logging.warning(f"No URLs found for {sitemap_url}")

        top_level_urls = []
        for url in urls:
            url = url.strip()
            if is_top_level(url, domain):
                top_level_urls.append(url)

        # Limit to 100 top-level URLs
        top_level_urls = top_level_urls[:100]

        output_file_path = os.path.join('to_scrape', f'{domain}_urls.txt')
        with open(output_file_path, 'w') as output_file:
            for top_level_url in top_level_urls:
                output_file.write(top_level_url + "\n")

        logging.info(f"Top-level URLs from {sitemap_url} written to {output_file_path}")

        download_folder = os.path.join('scraped', domain)
        try:
            download_pages(output_file_path, download_folder, timeout=30, default_skip=True, extensions=[], domains=[])
            logging.info(f"Downloaded pages for {domain} into {download_folder}")
        except Exception as e:
            logging.error(f"Failed to download pages for {domain}: {e}")

        root_directory = os.path.join('scraped', domain)
        for dirpath, dirnames, filenames in os.walk(root_directory):
            txt_files = [f for f in filenames if f.endswith('.txt')]

            if txt_files:
                full_paths = [os.path.join(dirpath, f) for f in txt_files]
                output_file = os.path.join(dirpath, 'filenames.txt')
                with open(output_file, 'w') as f:
                    for path in full_paths:
                        f.write(path + '\n')
                    try:
                        filter_files(output_file, threshold=0.5, gran='word', n=8, capacity=100000000, error_rate=1e-7, header=0, interval=1000000)
                        logging.info(f"Filtered files for directory {dirpath} using filter_files.")
                    except Exception as e:
                        logging.error(f"Failed to filter files in {dirpath}: {e}")
    except Exception as e:
        logging.critical(f"Critical error while processing {website_url}: {e}")

try:
    os.makedirs('websites', exist_ok=True)
    logging.info('Created "websites" directory.')

    os.makedirs('to_scrape', exist_ok=True)
    logging.info('Created "to_scrape" directory.')

    os.makedirs('scraped', exist_ok=True)
    logging.info('Created "scraped" directory.')

    with open('input_websites.txt', 'r') as infile:
        websites = infile.readlines()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_website, website_url): website_url for website_url in websites}

        for future in as_completed(futures):
            website_url = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Exception occurred while processing {website_url}: {e}")

except Exception as main_e:
    logging.critical(f"Critical error in the main script: {main_e}")
