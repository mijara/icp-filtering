import requests
import xml.etree.ElementTree as ET

def fetch_sitemap_urls(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        return response.content
    except requests.RequestException as e:
        print(f"Error fetching sitemap: {e}")
        return None

def extract_urls_from_sitemap(sitemap_url):
    sitemap_content = fetch_sitemap_urls(sitemap_url)
    if sitemap_content:
        urls = parse_sitemap(sitemap_content)
        return urls
    else:
        print("Failed to retrieve or parse sitemap.")
        return []

def parse_sitemap(content):
    urls = []
    root = ET.fromstring(content)
    for elem in root:
        if elem.tag.endswith('sitemap'):
            # It's a sitemap index
            for sitemap in elem:
                if sitemap.tag.endswith('loc'):
                    sub_sitemap_url = sitemap.text
                    sub_sitemap_content = fetch_sitemap_urls(sub_sitemap_url)
                    if sub_sitemap_content:
                        urls.extend(parse_sitemap(sub_sitemap_content))
        elif elem.tag.endswith('url'):
            # It's a regular sitemap
            for url in elem:
                if url.tag.endswith('loc'):
                    urls.append(url.text)
    return urls
    

def traverse_sitemap(sitemap_url):
    urls = extract_urls_from_sitemap(sitemap_url)
    urls=[url.strip() for url in urls]
    urls=sorted(urls, key=lambda url: (len(url.split('/')), url))
    print(len(urls))
    return urls