"""..."""
import os
import json
import random
import asyncio
import logging
import nodriver
import pandas as pd
from nodriver import Tab, Browser
from src.utils.clean_csv import get_missing_albums

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
    Saves the last successful track index to a file. This is useful in case of any interruptions
    or errors during runtime, allowing us to know when was the progress cut off.
    """
    progress_data = {
        "last_index": last_index,
        "album_cache": album_cache
    }

    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress_data, f)


async def load_progress() -> tuple[int, dict]:
    """
    Loads the last successful track index from a file.
    """
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            data = json.load(f)
            last_index = data.get("last_index", 0)
            album_cache = data.get("album_cache", {})
            return last_index, album_cache
    return 0, {}


async def check_for_error(page: Tab) -> bool:
    """Checks if we've got an error when searching for our page."""
    try:
        # Attempt to select the element with a 1.5s timeout.
        await page.select("div[class=page_error_content]", timeout=random.uniform(3.5,5))
        return True    # There is an error, true!

    except Exception:
        # If any exception occurs, assume the element is not present
        return False    # There are no errors, false!


async def random_interactions(page: Tab):
    """
    This method attempts to mimic human interactions when accessing a page. We're mainly
    concerned with creating delays and scrolling through the page while data is retrieved.
    """

    # Scroll three times with random directions and distances
    for _ in range(random.randint(1,5)):
        choice     = random.choice([True, False])
        scroll_val = round(random.uniform(50, 200), 3)

        # Scroll up or down depending on the random values previously acquired.
        scroll     = page.scroll_down if choice else page.scroll_up
        await scroll(scroll_val)

        # Introduce a short delay between scrolls to mimic reading time
        await asyncio.sleep(random.uniform(0.5, 2))


async def retrieve_rym_data(browser: Browser, artist_name: str, album_name: str) -> dict[str]:
    """
    ...
    """

    # Define constants and other vars.
    results         =  {}
    span_class_dict =  {
                        'pri_genres' : 'release_pri_genres',
                        'sec_genres' : 'release_sec_genres',
                        'pri_desc'   : 'release_pri_descriptors'
                        }
    release_type_list = ['album', 'ep', 'mixtape', 'comp']

    for release_type in release_type_list:

        # Access the page
        url   = f"https://rateyourmusic.com/release/{release_type}/{artist_name}/{album_name}/"
        page  = await browser.get(url)
        logging.info("Accessing URL: %s", url)


        # Check for errors...
        error = await check_for_error(page)
        if error:
            await asyncio.sleep(random.uniform(1.5,3))
            continue                    # If there is an error, try the next release type
        await random_interactions(page) # Otherwise, randomize interactions while acquiring data.


        # Commence gathering the data...
        try:    # NOTE : Susceptible to CAPTCHA taking longer than 10s.
            for span_name, span_class in span_class_dict.items():
                element  = await page.select(f"span[class={span_class}]")
                all_data = element.text_all

                logging.info("Successfully retreived data from RYM: %s", span_class)
                results[span_name] = [data.strip().lower() for data in all_data.split(",")]


        except Exception as e:
            # If any span fails, we skip this release type altogether.
            logging.error("Failed to retrieve data for release_type='%s'. Reason: %s",
                          release_type, e)
            continue

        # We can only possibly get here if there were no exceptions
        # when acquiring the data. Meaning the retrieval succeeded.
        break

    return results


async def retrieve_all_albums_data(album_list: list[str], recount: bool = False) -> list[dict[str]]:
    """
    ...

    Args:
        albums (dict[str, str]): _description_

    Returns:
        list[dict[str,str]]: _description_
    """
    results = []
    browser = await nodriver.start()

   # Load previous progress, if available.
    start_index, album_cache = await load_progress()
    album_request_counter    = 0
    total_albums             = len(album_list)
    start_index              = 0 if recount else start_index

    for i in range(start_index, total_albums):
        album_attributes      = album_list[i]
        artist_name           = album_attributes['query_artist']
        album_name            = album_attributes['query_album']

        album_key             = f"{artist_name}/{album_name}"
        album_request_counter += 1

        if album_key in album_cache:
            logging.info("Using cached data for album: %s", album_key)
            album_data = album_cache[album_key]

        else:
            album_data = await retrieve_rym_data(browser, artist_name, album_name)
            if not album_data:  # If no data was retrieved, there must've been failure. Thus, skip.
                logging.warning("Could not find data for %s : %s", artist_name, album_name)
                continue

            album_cache[album_key] = album_data

        # Save the data we've acquired
        results.append({
                        "artist_name": artist_name,
                        "album_name" : album_name,
                        "data"       : album_data
                       })
        await save_progress(i, album_cache)

        # Every 10 requests, take a break of 20-30 seconds.
        if album_request_counter % 10 == 0:
            long_break = random.uniform(20, 30)
            logging.info("Taking a long break for %d seconds after %d request",
                         long_break, album_request_counter)
            await asyncio.sleep(long_break)


    # If we've managed to get all the data without any errors,
    # write it in the `RESULT_FILE` to finish off.
    with open(RESULT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    logging.info("All album data successfully retrieved")
    return results


if __name__ == '__main__':
    csv_path   = '/Users/nico/Desktop/CIIC/CAPSTONE/essentia_demo/grouped_output_750_v2_clean.csv'
    json_path  = '/Users/nico/Desktop/CIIC/CAPSTONE/essentia_demo/progress.json'
    track_list = get_missing_albums(csv_path=csv_path, progress_path=json_path)

    logging.info("Starting scraping process")
    nodriver.loop().run_until_complete(retrieve_all_albums_data(track_list, recount=True))
    logging.info("Scraping process completed")


# NOTE : Also missing albums from artists which were featured twice...


"""MISSED :
Could not find data for aap-rocky : testing
Could not find data for aphex-twin : selected-ambient-works-8592
Could not find data for arcángel : feliz-navidad
Could not find data for asunojokei : island
Could not find data for at-the-drivein : relationship-of-command
Could not find data for beyoncé : lemonade
Could not find data for bizarrap-milo-j : en-dormir-sin-madrid
Could not find data for björk : post
Could not find data for camilasin-bandera : 4-latidos-tour-en-vivo
Could not find data for carlos-vives : corazón-profundo
Could not find data for cursive : cursives-domestica
Could not find data for dreamville-and-j-cole : revenge-of-the-dreamers-iii-directors-cut
Could not find data for eladio-carrión : 3men2-kbrn
Could not find data for evilgiane : evilslime
Could not find data for freddie-gibbs : bandana
Could not find data for harto-falión : im_my_best_friend
Could not find data for hic : l3ft-4-d3ad
Could not find data for jai-paul : leak-0413
Could not find data for jay-wheeler : platonicos
"""