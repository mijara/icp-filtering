import glob
import hashlib
import html
import http
import os
import logging
import re
import socket
import ssl
import time
import urllib.request

import requests
import tldextract

from cleaner import *
from utils import *

dir_path = os.path.dirname(os.path.realpath(__file__))


def exists(url):
    request = requests.get(url)
    return request.status_code == 200



def get_id_aus(link):
    id_ = link[link.rfind('/') + 1:link.rfind('.')]
    if id_[-1] == 'h':
        return id_[:-1]
    return id_

def to_skip(link, extensions=None, domains=None):
    """ domains can be:
            - just the name (as in: google)
            - main domain (as in: google.com)
            - subdomain (as in: news.google.com)
    """
    for ext in extensions:
        if link.endswith(ext):
            return True
    raw_url = get_raw_url(link)
    subdomain, domain, suffix = tldextract.extract(link)
    if domain in domains:
        return True
    if '.'.join([domain, suffix]) in domains:
        return True
    if '.'.join([subdomain, domain, suffix]) in domains:
        return True
    return False


def download_page(link, context=None, timeout=10, retries=3, backoff_factor=0.3):
    """
    Return code, page
    0: successfully read (write to index)
    1: bad_url (write to bad_url)
    2: unicode error (write to non_ascii_urls)
    3. bad_connection_urls

    When code is not 0, return ''
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for attempt in range(retries):
        try:
            req = urllib.request.Request(link, headers=headers)
            response = urllib.request.urlopen(req, context=context, timeout=timeout)
            page = response.read()
            return 0, page
        
        except (ValueError, urllib.error.HTTPError, urllib.error.URLError, http.client.HTTPException) as e:
            logging.warning(f'Error {e} for {link}')
            return 1, ''
        
        except UnicodeError as e:
            logging.warning(f'UnicodeError for {link}')
            return 2, ''
        
        except (ConnectionResetError, http.client.RemoteDisconnected, ConnectionError, socket.timeout, ssl.SSLError) as e:
            logging.warning(f'ConnectionError or Timeout on attempt {attempt+1} for {link}')
            if attempt < retries - 1:
                time.sleep(backoff_factor * (2 ** attempt))  # exponential backoff
                continue
            return 3, ''
        
        except Exception as e:
            logging.error(f'Unexpected error: {e} for {link}')
            return 1, ''

    return 1, ''



def get_current_idx(index_file, links):
    lines = open(index_file, 'r').readlines()
    idx = len(lines)
    if idx > 0:
        last_seen = lines[-1].strip()
        while True:
            link = links.readline().strip()
            if link == last_seen:
                break
    return idx, links


def download_pages(link_file,
                   folder,
                   timeout=30,
                   default_skip=False,
                   extensions=[],
                   domains=[]):
    """
    link_file (str):
        file contains links to pages to crawl. Each line contains one URL.
    folder (str):
        folder that you want to contain your downloaded pages.
    timeout:
        seconds to wait for a page to respond before abandoning it.

    default_skip (bool):
        True if you want to automatically skip all URLs that contain
        domains and extensions known to be scraper-unfriendly or NSFW.
        See the list of excluded domains at lazynlp/exclude_domains.txt.

        domains can be:
            - just the name (as in: google)
            - main domain (as in: google.com)
            - subdomain (as in: news.google.com)

        See the list of excluded extensions at
        lazynlp/exclude_extensions.txt

        You can also add your own domains and extensions to skip with domains
        and extensions and arguments.

    In the folder:
            Each URL is downloaded into a file, indexed by the order in which
            it is downloaded.
            The first line of each file is the URL.
            The rest is the textual content of the page.

            index.urls contains all the URLs that have been successfully downloaded.
            bad.urls contains the URLs that are bad.
            connection.urls contains the URLs that haven't been downloaded because
                            of connection issues.
            non_ascii.urls contains the URLs that haven't been downloaded because
                            of bad encoding issues.
            empty.urls contains the URLs that have empty textual content.
    """
    index_file = os.path.join(folder, 'index.urls')
    idx = 0
    links = open(link_file, 'r')

    if os.path.isdir(folder) and os.path.exists(index_file):
        """ If index file exists, we've downloaded from this list of
        URLs before, continue from where it left off the last time.
        """
        idx, links = get_current_idx(index_file, links)
        print(idx)
    else:
        os.makedirs(folder, exist_ok=True)

    index = open(os.path.join(folder, 'index.urls'), 'a')
    skipped_urls = open(os.path.join(folder, 'skip.urls'), 'a')
    bad_connection_urls = open(os.path.join(folder, 'connection.urls'), 'a')
    bad_urls = open(os.path.join(folder, 'bad.urls'), 'a')
    non_ascii_urls = open(os.path.join(folder, 'non_ascii.urls'), 'a')
    empty_urls = open(os.path.join(folder, 'empty.urls'), 'a')

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    hashed = hashlib.sha1()

    if default_skip:
        ext_lines = open(f'{dir_path}/exclude_extensions.txt', 'r').readlines()
        extensions.extend([line.strip() for line in ext_lines])
        domain_lines = open(f'{dir_path}/exclude_domains.txt', 'r').readlines()
        domains.extend([line.strip() for line in domain_lines])

    for link in links:
        link = link.strip()
        # if to_skip(link, extensions, domains):
        #     skipped_urls.write(link + '\n')
        #     print('Skip', link)
        #     continue

        code, page = download_page(link, ctx, timeout)
        if code == 1:
            bad_urls.write(link + '\n')
        elif code == 2:
            non_ascii_urls.write(link + '\n')
        elif code == 3:
            bad_connection_urls.write(link + '\n')
        if code > 0:
            continue

        txt = clean_page(page)

        if not txt:
            print('Empty page', link)
            empty_urls.write(link + '\n')
            continue

        print(idx, link)
        hashed.update(str(time.time()).encode())
        name = hashed.hexdigest()
        with open(f'{folder}/{idx}_{name}.txt', 'w') as out:
            out.write(link + '\n' + txt)

        print(find_unprintable(txt))
        index.write('{}\n'.format(link))
        idx += 1

    links.close()
