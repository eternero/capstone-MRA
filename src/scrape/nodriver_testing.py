"""..."""
import os
import csv
import json
import random
import asyncio
import logging
import nodriver
import pandas as pd
from nodriver import Tab, Browser
from src.utils.clean_csv import get_query_columns

# Configuring the logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='scraping.log',
    filemode='a'
)

PROGRESS_FILE = "progress.csv"

class NoDriving:
    """Object whose purpose is to scrape data in an undetected manner from websites. For us, our
    target is RateYourMusic (RYM) - from which we will be getting User-Generated Content (UGC) such
    as Primary Genre, Secondary Genre and Album Descriptors.

    This object is concerned with only two things really
        1. Scraping the data from RateYourMusic.
        2. Saving and Loading that very same data.

    Attributes:
        dataset_path  : The path to the csv file that contains the albums to be searched for. It
                        must only have the columns `ARTIST` and `ALBUM`. It does not need to be
                        cleaned either, this class will take care of doing so.
        progress_path : The path pointing to the csv file in which we will save the RYM data. If one
                        does not exist, it will be created.
        album_df      : The DataFrame which contains all of our relevant album data.

    DataFrames:
        album_df      : The main DataFrame to be used. It is extracted from `dataset_path` and then
                        cleaned and normalized - we also compare it to whatever we've gone over in
                        the `progress_path` to avoid repetition. This will be used to iterate over
                        the albums for which we are yet to acquire their data.
        progress_df   : The DataFrame used to represent our previously acquired data. This is only
                        briefly used, since its only utility is to allow us to intuitively compare
                        it with the `album_df` to figure out what albums remain to be retrieved.
    """

    def __init__(self, dataset_path : str, progress_path : str):
        self.dataset_path  = dataset_path
        self.progress_path = progress_path

        # Define constants.
        self.attributes    = [
                                "artist", "album",
                                "pri_genres", "sec_genres", "pri_desc"
                             ]

        self.span_classes  = {
                                'pri_genres' : 'release_pri_genres',
                                'sec_genres' : 'release_sec_genres',
                                'pri_desc'   : 'release_pri_descriptors'
                             }

        self.release_types = [
                                'album', 'ep', 'mixtape', 'comp'
                             ]

        # Initialize our dataframes.
        progress_exists    = os.path.exists(progress_path) and not os.path.getsize(progress_path) == 0
        self.progress_df   = (
                                pd.read_csv(progress_path) if progress_exists
                                else pd.DataFrame(columns=self.attributes)
                             )

        self.album_df      = self.get_missing_albums()


    def save_progress(self, album_data : dict[str]):
        """
        This methods adds a new line to a CSV with data from RYM. If the CSV does not yet exist, it
        will take the task of initializing it with the columns:
            `artist, album, pri_genre, sec_genre, pri_desc`
        """

        # Open the file in append mode
        with open(self.progress_path, mode = "a", newline = "", encoding = "utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = self.attributes)

            # If the file does not exist or is empty, write the header.
            if not os.path.exists(self.progress_path) or os.path.getsize(self.progress_path) == 0:
                writer.writeheader()

            # Write the data into file.
            writer.writerow(album_data)


    def get_missing_albums(self):
        """
        Compares the `album_df` against the data in the .csv located at `progress_path`. It will
        update the album list to only include those albums whose data has not been retrieved yet.

        If there does not exist any .csv at `progress_path` yet, then it will simply return the
        entirety of `album_df` unaltered.
        """
        album_df            = get_query_columns(self.dataset_path)
        progress_pairs      = set(zip(self.progress_df["album"], self.progress_df["artist"]))

        # Get the mask which will indicate all of the missing albums
        remainder_mask      = [
                                (row["query_album"], row["query_artist"]) not in progress_pairs
                                for _, row in album_df.iterrows()
                              ]

        return album_df[remainder_mask]


    @staticmethod
    async def check_for_error( page: Tab) -> bool:
        """Checks if we've got an error when searching for our page."""
        try:
            # Attempt to select the element with a 1.5s timeout.
            await  page.select("div[class=page_error_content]", timeout=random.uniform(3.5,5))
            return True     # There is an error, true!

        except Exception:
            # If any exception occurs, assume the element is not present
            return False    # There are no errors, false!


    @staticmethod
    async def random_interactions(page: Tab):
        """
        This method attempts to mimic human interactions when accessing a page. We're mainly
        concerned with creating delays and scrolling through the page while data is retrieved.
        """

        # Scroll three times with random directions and distances
        for _ in range(random.randint(2,6)):
            choice     = random.choice([True, False])
            scroll_val = round(random.uniform(50, 200), 3)

            # Scroll up or down depending on the random values previously acquired.
            scroll     = page.scroll_down if choice else page.scroll_up
            await scroll(scroll_val)

            # Introduce a short delay between scrolls to mimic reading time
            await asyncio.sleep(random.uniform(1, 2.5))


    async def retrieve_rym_data(self, browser: Browser,
                                artist_name  : str,
                                album_name   : str) -> dict[str]:
        """
        Retrives the data from RateYourMusic for one release (album, ep, mixtape or compilation).
        """

        results   =  {}
        for release in self.release_types:

            # Access the page.
            url   = f"https://rateyourmusic.com/release/{release}/{artist_name}/{album_name}/"
            page  = await browser.get(url)
            logging.info("Accessing URL: %s", url)


            # Check for errors.
            error = await NoDriving.check_for_error(page)
            if error:
                await asyncio.sleep(random.uniform(2.5, 5))
                continue                                # If there is an error, try the next release type
            await NoDriving.random_interactions(page)   # Otherwise, randomize interactions while acquiring data.

            # Commence gathering the data..
            try:    # NOTE : Susceptible to CAPTCHA taking longer than 10s.
                for span_name, span_class in self.span_classes.items():
                    element            = await page.select(f"span[class={span_class}]")
                    all_data           = element.text_all

                    logging.info("Successfully retreived data from RYM: %s", span_class)
                    results[span_name] = json.dumps([data.strip().lower()               # Not sure if
                                                     for data in all_data.split(",")])  # this will work...


            except Exception as e:
                # If any span fails, we skip this release type altogether.
                logging.error("Failed to retrieve data for release type ='%s'. Reason: %s",
                            release, e)
                continue

            # We can only possibly get here if there were no exceptions
            # when acquiring the data. Meaning the retrieval succeeded.
            break

        return results


    async def retrieve_all_albums_data(self):
        """
        Iterates through the entirety of `album_df` and attempts to retrieve all of the data
        from RateYourMusic for each of the albums in this DataFrame. As it proceeds, it will
        continually save the progress.
        """
        # Start up the browser!
        browser = await nodriver.start()

        for row in self.album_df.itertuples(index=True):
            artist_name = row.query_artist
            album_name  = row.query_album

            album_data  = await self.retrieve_rym_data(browser, artist_name, album_name)
            if not album_data:  # If no data was retrieved, there must've been failure. Thus, skip.
                logging.warning("Could not find data for %s : %s", artist_name, album_name)
                continue

            # Save the data we've acquired
            retrieved_data = {"artist" : artist_name, "album" : album_name}
            retrieved_data.update(album_data)

            # Write it to the csv file at `self.progress_path`.
            self.save_progress(retrieved_data)

            # Every 10 requests, take a break of 20-30 seconds.
            if (row.Index + 1) % 10 == 0:
                long_break = random.uniform(20, 30)
                logging.info("Taking a long break for %d seconds after %d request",
                             long_break, row.Index)
                await asyncio.sleep(long_break)



if __name__ == '__main__':
    CSV_PATH   = '/Users/nico/Desktop/CIIC/CAPSTONE/essentia_demo/grouped_output_750_v2_clean.csv'
    no_driving = NoDriving(CSV_PATH, PROGRESS_FILE)

    logging.info("Starting scraping process")
    nodriver.loop().run_until_complete(no_driving.retrieve_all_albums_data())
    logging.info("Scraping process completed")
