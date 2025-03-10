import os
import json
import random
from librosa import ex
import nodriver
import logging
from pprint import pprint
import asyncio
from nodriver import Tab, Browser

# Configuring the logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='scraping.log',
    filemode='a'
)

PROGRESS_FILE = "progress.json"
RESULT_FILE = "rym_result.json"

async def save_progress(last_index: int, album_cache: dict):
    """
    ...
    Saves the last successful track index to a file.
    If we get blocked (or IP banned) we know where we at
    """
    progress_data = {
        "last_index": last_index,
        "album_cache": album_cache
    }

    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress_data, f)

async def load_progress() -> tuple[int, dict]:
    """
    ...
    Loads the last successful track index from a file.
    """
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            data = json.load(f)
            last_index = data.get("last_index", 0)
            album_cache = data.get("album_cache", {})
            return last_index, album_cache
    return 0, {}


async def human_like_delay():
    """
    ...

    Function to mimics random human interactions
    Maybe add more time every 10-20 minutes
    """
    base_delay = random.uniform(1, 3)
    random_movement = random.uniform(0.1, 0.5)
    await asyncio.sleep(base_delay + random_movement)


async def random_interactions(page: Tab):
    """
    ...
    Trying to mimic random human interactions
    """

    # 70% chance of scrolling randomly
    if random.random() < 0.7:
        scroll_amount  = random.randint(1, 5)
        if random.random() < 0.5:
            await page.scroll_down(scroll_amount)
        else:
            await page.scroll_up(scroll_amount)
        await human_like_delay()

async def retrieve_rym_data(page : Tab, span_class:str,) -> list[str]:
    """
    ...

    Args:
        span_class (str): _description_

    Returns:
        list[str]: _description_
    """
    try:
        await random_interactions(page)
        element = await page.select(f"span[class={span_class}]")
        all_data = element.text_all
        logging.info("Successfully retreived data from RYM: %s", span_class)
        return [data.strip().lower() for data in all_data.split(",")]

    except Exception as e:
        logging.error("Failed to retrieve data from RMY, span class %s: %s", span_class, str(e))
        return []



async def retrieve_album_data(browser: Browser, artist_name: str, album_name: str):
    """
    ...

    Args:
        artist_name (str): _description_
        album_name  (str): _description_
    """
    # Define constants
    span_classes =  {
                      'pri_genres' : 'release_pri_genres',
                      'sec_genres' : 'release_sec_genres',
                      'pri_desc'   : 'release_pri_descriptors'
                    }

    # Define dictionary to save shit
    results = {}

    # Access the page
    url = f"https://rateyourmusic.com/release/album/{artist_name}/{album_name}/"
    logging.info("Accessing URL: %s", url)

    try:    
        page = await browser.get(url)

        # Get all of our beautiful data
        for span_name, span_class in span_classes.items():
            span_data = await retrieve_rym_data(page, span_class)
            results[span_name] = span_data

        logging.info("Successfully retreived data for album: %s - %s", artist_name, album_name)
        print(artist_name,album_name)
        pprint(results)
        print("\n" + "-" * 50 + "\n")
        return results
    
    except Exception as e:
        logging.error("Error retreiving album data for %s - %s: %s", artist_name, album_name, str(e))
        return {}

async def retrieve_all_albums_data(album_dict: dict[str, str]) -> list[dict[str,str]]:
    """
    ...

    Args:
        albums (dict[str, str]): _description_

    Returns:
        list[dict[str,str]]: _description_
    """
    results = []
    browser = await nodriver.start()
    # Load previous progress, if avaylable, if this fuckers ban us
    start_index, album_cache = await load_progress()
    album_request_counter = 0 

    album_items = list(album_dict.items())
    total_albums = len(album_items)

    for i in range(start_index, total_albums):
        artist_name, album_name = album_items[i]
        album_request_counter += 1
        album_key = f"{artist_name}/{album_name}"
        if album_key in album_cache:
            logging.info("Using cached data for album: %s", album_key)
            album_data = album_cache[album_key]
        else:
            album_data = await retrieve_album_data(browser, artist_name, album_name)
            album_cache[album_key] = album_data
    
        results.append({
                "artist_name": artist_name,
                "album_name": album_name,
                "data": album_data
        })

        await save_progress(i, album_cache)

        sleep_time = random.randint(2,4) + random.randint(1,1000) / 1000
        logging.info("Sleeping for %.2f seconds before next request", sleep_time)
        await asyncio.sleep(sleep_time)

        if album_request_counter % 20 == 0:
            long_break = random.randint(10, 30)
            logging.info("Taking a long break for %d seconds after %d request", long_break, album_request_counter)
            await asyncio.sleep(long_break)
    
    with open(RESULT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    logging.info("All album data successfully retrieved")
    return results


if __name__ == '__main__':

    a_dict = {
        'bladee': 'red-light',
        'ecco2k': 'e',
        'burial': 'untrue',
        "adele": "30",
        "billie-eilish": "when-we-all-fall-asleep-where-do-we-go",
    }

    logging.info("Starting scraping process")
    nodriver.loop().run_until_complete(retrieve_all_albums_data(a_dict))
    logging.info("Scraping process completed")
    
   