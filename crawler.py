# crawler.py

import os

import requests


def fetch_docs(url_list, output_dir="data/raw"):
    """
    Fetches HPC documentation from a list of URLs and saves them locally.

    :param url_list: List[str] - A list of URLs pointing to HPC documentation (HTML, PDF, etc.).
    :param output_dir: str - The directory where downloaded files will be stored.

    :return: List[str] - A list of file paths to the locally saved documents.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []

    for url in url_list:
        # Infer a filename from the URL (use 'index.html' if the URL ends with '/')
        filename = url.rsplit('/', 1)[-1] or "index.html"
        local_path = os.path.join(output_dir, filename)

        try:
            # Download the file with a reasonable timeout
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raises an HTTPError if the status is not 200

            # Write the content to a local file in binary mode
            with open(local_path, "wb") as f:
                f.write(response.content)

            saved_paths.append(local_path)

        except requests.exceptions.RequestException as e:
            # Handle any network or HTTP errors gracefully
            print(f"Error fetching {url}: {e}")

    return saved_paths



if __name__ == "__main__":
    print("Fetching HPC documentation...")
    url_list = ['https://wiki.u-gov.it/confluence/display/SCAIUS/HPC+at+CINECA%3A+User+Documentation']
    fetch_docs(url_list)
