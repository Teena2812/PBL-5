import os
import re
import csv
import xml.etree.ElementTree as ET
import urllib.request
import urllib.parse
import time

# Define the folder structure and search queries for each theme
THEMES = {
    "01_federated_learning_fundamentals": {
        "query": 'all:"Federated Learning" AND (all:survey OR all:fundamentals OR all:overview)',
        "max_results": 13
    },
    "02_fl_in_healthcare": {
        "query": 'all:"Federated Learning" AND (all:healthcare OR all:clinical OR all:medical OR all:disease)',
        "max_results": 13
    },
    "03_personalized_fl": {
        "query": 'all:"Personalized Federated Learning" OR all:FedProx OR (all:"Federated Learning" AND all:non-IID)',
        "max_results": 12
    },
    "04_explainable_ai_clinical": {
        "query": '(all:"Explainable AI" OR all:SHAP OR all:LIME) AND (all:clinical OR all:medical OR all:healthcare OR all:disease)',
        "max_results": 12
    }
}

BASE_DIR = "research_papers"
INDEX_FILE = os.path.join(BASE_DIR, "papers_index.csv")

def clean_filename(title):
    # Remove math symbols, LaTeX, punctuation and keep it clean
    cleaned = re.sub(r'\s+', ' ', title)
    cleaned = re.sub(r'[^a-zA-Z0-9\s\-_]', '', cleaned)
    return cleaned.strip()[:120]

def search_arxiv(query, max_results):
    encoded_query = urllib.parse.quote(query)
    url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&max_results={max_results}&sortBy=relevance"
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
        return xml_data
    except Exception as e:
        print(f"Error fetching from arXiv for query '{query}': {e}")
        return None

def parse_and_download(xml_data, theme_folder):
    if not xml_data:
        return []
    
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    try:
        root = ET.fromstring(xml_data)
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return []
    
    papers = []
    entries = root.findall('atom:entry', ns)
    
    os.makedirs(os.path.join(BASE_DIR, theme_folder), exist_ok=True)
    
    for entry in entries:
        try:
            title = entry.find('atom:title', ns).text
            title = " ".join(title.split()) # Clean up multiple whitespaces/newlines
            
            summary = entry.find('atom:summary', ns).text
            summary = " ".join(summary.split())
            
            published = entry.find('atom:published', ns).text
            year = published.split('-')[0]
            
            authors = [author.find('atom:name', ns).text.strip() for author in entry.findall('atom:author', ns)]
            authors_str = ", ".join(authors)
            
            html_link = ""
            pdf_link = ""
            for link in entry.findall('atom:link', ns):
                rel = link.attrib.get('rel')
                title_attr = link.attrib.get('title')
                href = link.attrib.get('href')
                if rel == 'alternate':
                    html_link = href
                elif title_attr == 'pdf' or link.attrib.get('type') == 'application/pdf':
                    pdf_link = href
            
            if not html_link and pdf_link:
                html_link = pdf_link.replace('/pdf/', '/abs/')
                
            safe_title = clean_filename(title)
            pdf_filename = f"{safe_title}.pdf"
            pdf_path = os.path.join(BASE_DIR, theme_folder, pdf_filename)
            
            pdf_downloaded = "N"
            
            if pdf_link:
                print(f"PDF Link available: {pdf_link}")
                pdf_downloaded = "Y"
            
            papers.append({
                "title": title,
                "authors": authors_str,
                "year": year,
                "source": "arXiv",
                "link": html_link or pdf_link,
                "theme": theme_folder,
                "pdf_available": pdf_downloaded,
                "abstract": summary
            })
        except Exception as e:
            print(f"Error processing entry: {e}")
            
    return papers

def main():
    print("Starting Paper Collection Script...")
    all_papers = []
    
    os.makedirs(BASE_DIR, exist_ok=True)
    
    for theme_folder, info in THEMES.items():
        print(f"\nProcessing Theme: {theme_folder}")
        xml_data = search_arxiv(info["query"], info["max_results"])
        theme_papers = parse_and_download(xml_data, theme_folder)
        all_papers.extend(theme_papers)
        # Polite delay to prevent arXiv rate limiting
        time.sleep(2)
        
    print(f"\nWriting index to {INDEX_FILE}...")
    with open(INDEX_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["title", "authors", "year", "source", "link", "theme", "pdf_available", "abstract"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for paper in all_papers:
            writer.writerow(paper)
            
    # Calculate statistics
    total = len(all_papers)
    pdf_count = sum(1 for p in all_papers if p["pdf_available"] == "Y")
    theme_counts = {}
    for p in all_papers:
        theme_counts[p["theme"]] = theme_counts.get(p["theme"], 0) + 1
        
    print("\n" + "="*40)
    print("COLLECTION SUMMARY")
    print("="*40)
    print(f"Total papers found: {total}")
    print(f"PDFs downloaded successfully: {pdf_count}")
    print(f"Citation-only papers: {total - pdf_count}")
    print("\nPapers per theme:")
    for theme, count in theme_counts.items():
        print(f" - {theme}: {count}")
    print("="*40)

if __name__ == "__main__":
    main()
