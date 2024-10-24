import os
import pandas as pd
import re

import os
import pandas as pd

def load_and_concatenate_files(directory):
    # List to hold final data
    data = []

    # Crawl through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        subdirectory_name = os.path.basename(root)
        # Check if subdirectory has .txt files that begin with numeral_
        txt_files = [f for f in files if f.endswith('.txt') and f.split('_')[0].isdigit()]

        if txt_files:
            clean_files_list_path = os.path.join(root, 'clean_files.list')

                # Check if clean_files.list exists in the subdirectory
            if os.path.exists(clean_files_list_path):
                with open(clean_files_list_path, 'r') as list_file:
                    clean_files = list_file.read().splitlines()

                # Filter and sort the files by the numeric prefix
                clean_files = [f for f in clean_files if os.path.basename(f).split('_')[0].isdigit() and os.path.basename(f) != 'filenames.txt']
                clean_files.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
                clean_files = clean_files[:25]
                concatenated_content = ""

                # Load and concatenate the contents of all files listed in clean_files.list
                for clean_file_path in clean_files:
                    # Prepend the root directory to the relative paths in clean_files.list
                    full_file_path = os.path.join(directory, clean_file_path)
                    if os.path.exists(full_file_path):
                        with open(full_file_path, 'r') as f:
                            content = f.read().strip()
                            concatenated_content += content + "\n"
                            if not content:
                                print(f"Warning: {full_file_path} is empty.")
                    else:
                        print(f"Warning: {full_file_path} does not exist.")

                    # Append the subdirectory name and concatenated content to the data list
                if concatenated_content.strip():
                    data.append({
                            'website': subdirectory_name,
                            'content': concatenated_content.strip(),
                            'len': len(concatenated_content.strip())
                        })
                else:
                    print(f"Warning: No content to concatenate in {subdirectory_name}.")

    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    df.to_csv("iiq.csv")
    return df

def is_valid_url(url):
    if isinstance(url, str):
        regex = re.compile(
            r'^(https?|ftp)://'  # http:// or https:// or ftp://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
            r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return re.match(regex, url) is not None
    return False

def clean_url(url):
    if isinstance(url, str):
        # Remove the http://, https://, and www.
        url = re.sub(r'^https?://(www\.)?', '', url)
        # Remove the top-level domain (TLD) codes like .com, .org, etc.
        url = re.sub(r'\.[a-z]{2,6}$', '', url)
        # Remove any trailing slashes
        url = url.rstrip('/')
        return url

def main():
    directory_path = '/Users/akshaymijar/icp-filtering' 
    websites_df = load_and_concatenate_files(directory_path)
    print("Loaded website data into dataframe.")
    #print(websites_df.info())
    leads_df=pd.read_csv("/Users/akshaymijar/icp-filtering/rb2b30cleaned.csv")
    filtered_df = leads_df[leads_df['Website'].apply(is_valid_url)]
    filtered_df = filtered_df[filtered_df['LinkedInUrl'].apply(is_valid_url)]
    columns_subset = ['LinkedInUrl','Title','Website','CompanyName']
    final_df = filtered_df[columns_subset]
    leads_final_df=final_df.copy()
    leads_final_df['website'] = leads_final_df['Website'].apply(clean_url)
    print("Loaded leads sheet, cleaned up the website urls to join on.")
    print(leads_final_df.info())
    downloaded_websites=set(sorted(websites_df["website"].to_list()))
    lead_websites=set(sorted(leads_final_df["website"].to_list()))
    print(f"No. of downloaded websites: {len(downloaded_websites)} No. of lead websites: {len(lead_websites)}")
    leads_final_df = leads_final_df[leads_final_df['website'].isin(downloaded_websites)]
    result_df = pd.merge(leads_final_df, websites_df, on='website', how='left')
    result_df.to_excel('lead_website_contents_newrun.xlsx', index=False)
    print("Lead website sheets concatenated with website content")
    
if __name__ == "__main__":
    main()