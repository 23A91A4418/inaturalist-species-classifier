#!/usr/bin/env python3
"""
scripts/01_fetch_data.py
Data Acquisition Script for iNaturalist Species Classification Pipeline.
Downloads images for top species within a specified taxonomic order from iNaturalist API.
"""

import os
import sys
import time
import logging
import requests
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

load_dotenv()

BASE_API_URL = "https://api.inaturalist.org/v1"
TAXON_ID = int(os.getenv("TAXON_ID", 47157))  # 47157 = Lepidoptera (Butterflies & Moths)
NUM_SPECIES = int(os.getenv("NUM_SPECIES", 10))
MIN_OBSERVATIONS = int(os.getenv("MIN_OBSERVATIONS", 200))
TARGET_IMAGES = int(os.getenv("IMAGES_PER_SPECIES", 200))
RAW_DATA_DIR = Path("data/raw")


def fetch_top_species(taxon_id: int, target_num: int, min_obs: int):
    """Fetch top species within a taxon order that have at least min_obs observations with photos."""
    endpoint = f"{BASE_API_URL}/observations/species_counts"
    params = {
        "taxon_id": taxon_id,
        "quality_grade": "research",
        "has_photos": "true",
        "per_page": 100
    }
    
    logging.info(f"Querying species counts for taxon ID: {taxon_id}...")
    headers = {"User-Agent": "iNaturalistSpeciesClassifier/1.0"}
    
    try:
        response = requests.get(endpoint, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logging.error(f"Failed to fetch species counts from iNaturalist API: {e}")
        sys.exit(1)
        
    species_list = []
    for item in data.get("results", []):
        count = item.get("count", 0)
        taxon = item.get("taxon", {})
        rank = taxon.get("rank")
        name = taxon.get("name")
        spec_id = taxon.get("id")
        
        if rank == "species" and name and spec_id and count >= min_obs:
            species_list.append({
                "id": spec_id,
                "name": name,
                "count": count
            })
            if len(species_list) >= target_num:
                break
                
    logging.info(f"Selected {len(species_list)} species with >= {min_obs} observations.")
    return species_list


def download_single_image(image_url: str, img_path: Path, headers: dict) -> bool:
    """Download a single image file to img_path."""
    if img_path.exists():
        return True
    try:
        resp = requests.get(image_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            with open(img_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception:
        pass
    return False


def download_images_for_species(species_info: dict, target_images_count: int, save_dir: Path):
    """Download images for a given species up to target_images_count using thread pool."""
    species_name = species_info["name"]
    species_id = species_info["id"]
    species_dir = save_dir / species_name
    species_dir.mkdir(parents=True, exist_ok=True)
    
    existing_count = len(list(species_dir.glob("*.jpg")))
    if existing_count >= target_images_count:
        logging.info(f"[{species_name}] Already has {existing_count} images (>= {target_images_count}). Skipping.")
        return
        
    needed = target_images_count - existing_count
    logging.info(f"[{species_name}] Currently has {existing_count} images. Fetching {needed} more...")
    
    page = 1
    image_tasks = []
    headers = {"User-Agent": "iNaturalistSpeciesClassifier/1.0"}
    
    while (existing_count + len(image_tasks)) < target_images_count and page <= 10:
        endpoint = f"{BASE_API_URL}/observations"
        params = {
            "taxon_id": species_id,
            "quality_grade": "research",
            "photos": "true",
            "per_page": 100,
            "page": page,
            "order": "desc",
            "order_by": "votes"
        }
        
        try:
            resp = requests.get(endpoint, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            obs_data = resp.json()
        except Exception as e:
            logging.error(f"Error fetching page {page} for {species_name}: {e}")
            time.sleep(1)
            page += 1
            continue
            
        observations = obs_data.get("results", [])
        if not observations:
            break
            
        for obs in observations:
            if (existing_count + len(image_tasks)) >= target_images_count:
                break
                
            obs_id = obs.get("id")
            photos = obs.get("photos", [])
            if not photos:
                continue
                
            photo_url = photos[0].get("url")
            if not photo_url:
                continue
                
            medium_url = photo_url.replace("square.jpg", "medium.jpg").replace("square.jpeg", "medium.jpeg").replace("/square.", "/medium.")
            img_path = species_dir / f"{obs_id}.jpg"
            if not img_path.exists():
                image_tasks.append((medium_url, img_path))
            
        page += 1
        time.sleep(0.3)
        
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_single_image, url, path, headers) for url, path in image_tasks]
        for future in as_completed(futures):
            future.result()
                
    final_count = len(list(species_dir.glob("*.jpg")))
    logging.info(f"Finished {species_name}: {final_count} total images in directory.")


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    species_list = fetch_top_species(TAXON_ID, NUM_SPECIES, MIN_OBSERVATIONS)
    
    for species in species_list:
        download_images_for_species(species, TARGET_IMAGES, RAW_DATA_DIR)
        time.sleep(0.3)
        
    logging.info("Data acquisition complete!")


if __name__ == "__main__":
    main()
