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
from src.utils import load_json, is_missing

# Configuring the logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='scraping.log',
    filemode='a'
)

class DataHandler:
    """
    This class will be in charge of handling the data that we acquire throughout the retrieval of
    User Generated Content from RateYourMusic. It's basic functionality includes saving and loading
    progress into CSVs.

    More extensive functionality includes analyzing saved data to determine any errors. Lastly, it
    will also be tasked with potentially deleting tracks from albums which don't have good data.
    For simplicity, this class will be inherited by `NoDriving`.

    NOTE : This method could also be in charged of joining the progress df with the dataset df.

    Attributes:
        dataset_path  : The path to the csv file that contains the albums to be searched for. It
                        must only have the columns `ARTIST` and `ALBUM`. It does not need to be
                        cleaned either, this class will take care of doing so.
        progress_path : The path pointing to the csv file in which we will save the RYM data. If one
                        does not exist, it will be created.
        failure_path  : The path pointing to the csv file used to record failures. Failures indicate
                        that an album was not successfully retrieved from RateYourMusic.
    """

    def __init__(self, dataset_path : str, progress_path : str, failure_path : str):
        self.dataset_path  = dataset_path
        self.progress_path = progress_path
        self.failure_path  = failure_path


    def save_progress(self, path : str, album_data : dict[str], headers : list[str]):
        """
        This methods adds a new line to a CSV with data from RYM. If the CSV does not yet exist, it
        will take the task of initializing it with the columns:
            `artist, album, pri_genre, sec_genre, pri_desc`
        """

        # Open the file in append mode
        with open(path, mode = "a", newline = "", encoding = "utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = headers)

            # If the file does not exist or is empty, write the header.
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                writer.writeheader()

            # Write the data into file.
            writer.writerow(album_data)


    def load_progress(self) -> pd.DataFrame:
        """
        Compares the `album_df` against the data in the .csv located at `progress_path`. It will
        update the album list to only include those albums whose data has not been retrieved yet.

        If there does not exist any .csv at `progress_path` yet, then it will simply return the
        entirety of `album_df` unaltered.

        NOTE : Now also checking for the `failure_path`. These are the albums which will have to
               be manually retrieved or are just of no good.
        """
        # Acquire our current DataFrame and then retrieve the progress.
        album_df           = pd.read_csv(self.dataset_path)
        album_df           = album_df.drop_duplicates(subset=['clean_artist', 'clean_album'])
        pairs_to_exclude   = set()

        # Check for progress.
        if os.path.exists(self.progress_path) and os.path.getsize(self.progress_path) > 0:
            progress_df    = pd.read_csv(self.progress_path)
            pairs_to_exclude.update(zip(progress_df["album"], progress_df["artist"]))

        # Check for failures. These will have to be added manually.
        if os.path.exists(self.failure_path)  and os.path.getsize(self.failure_path) > 0:
            failure_df     = pd.read_csv(self.failure_path)
            pairs_to_exclude.update(zip(failure_df["album"], failure_df["artist"]))

        # If any pairs are in either CSV, exclude them from the DataFrame.
        if pairs_to_exclude:
            remainder_mask = [
                               (row["clean_album"], row["clean_artist"]) not in pairs_to_exclude
                               for _, row in album_df.iterrows()
                             ]
            album_df = album_df[remainder_mask]

        # Only keep the important columns.
        return album_df[["clean_album", 'clean_artist']]


    def analyze_progress(self) -> pd.DataFrame:
        """
        Reads and analyzes the progress dataset. This is done with the purpose of finding albums
        which were unsuccessful in their retrieval of data from RateYourMusic.

        Returns:
            incomplete_df : The DataFrame which contains the missing pairs of `artist : album`.
        """

        # Read the CSV file
        progress_df = pd.read_csv(self.progress_path)

        # Convert JSON strings to lists for the target columns.
        for col in ['pri_genres', 'sec_genres', 'pri_desc']:
            progress_df[col] = progress_df[col].apply(load_json)

        # Create a mask for albums missing any of the three fields.
        missing_mask         = (
            progress_df['pri_genres'].apply(is_missing) |
            progress_df['sec_genres'].apply(is_missing) |
            progress_df['pri_desc'].apply(is_missing)
        )

        # Filter DataFrame to keep only artist and album for albums with missing fields.
        incomplete_df        = progress_df.loc[missing_mask, ['artist', 'album']]

        # Rename columns since this will be concatenated back to the `album_df`
        return incomplete_df.rename(columns={'artist' : 'clean_artist', 'album' : 'clean_album'})



class NoDriving(DataHandler):
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
    """

    def __init__(self, dataset_path : str, progress_path : str, failure_path : str):
        super().__init__(dataset_path, progress_path, failure_path)

        # Define constants.
        self.fail_headers  = [
                                "artist", "album"
                             ]

        self.album_headers = [
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

        # Initialize our dataframe. First loading the progress and then adding any missing albums.
        progress_df        = self.load_progress()
        missing_df         = self.analyze_progress()
        self.album_df      = pd.concat([progress_df, missing_df])

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

        for release in self.release_types:

            # Access the page.
            url     = f"https://rateyourmusic.com/release/{release}/{artist_name}/{album_name}/"
            page    = await browser.get(url)
            results =  {}
            logging.info("Accessing URL: %s", url)

            # Check for errors.
            error = await NoDriving.check_for_error(page)
            if error:
                await asyncio.sleep(random.uniform(2.5, 5))
                continue                                # If there is an error, try the next release type
            await NoDriving.random_interactions(page)   # Otherwise, randomize interactions while acquiring data.

            # Commence gathering the data..
            try:
                for span_name, span_class in self.span_classes.items():
                    element            = await page.select(f"span[class={span_class}]")
                    all_data           = element.text_all

                    logging.info("Successfully retreived data from RYM: %s", span_class)

                    extracted_data     = [data.strip().lower()for data in all_data.split(",")]
                    results[span_name] = json.dumps(extracted_data)

                    # This is done since at times, an attribute will be in the page source code,
                    # but it won't really have anything (e.g. descriptors = ['']) and that's no good
                    if is_missing(extracted_data) or not extracted_data:
                        logging.warning("No %s data could be extracted for release : %s",span_class,
                                                                                         album_name)
                        results = {}    # Set results to `{}` since we don't want incomplete data.
                        break

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

            # Get our basic fields
            artist_name    = row.clean_artist
            album_name     = row.clean_album

            # These fields determine what data will be saved (failure or progress) and where.
            save_path      = self.progress_path
            headers        = self.album_headers

            # Now we're ready to retrieve our data.
            retrieved_data = {"artist" : artist_name, "album" : album_name}
            album_data     = await self.retrieve_rym_data(browser, artist_name, album_name)
            retrieved_data.update(album_data)

            if not album_data:  # If no data was retrieved, record the album as a failure.
                save_path  = self.failure_path
                headers    = self.fail_headers
                logging.warning("Could not find data for %s : %s", artist_name, album_name)

            # Write the current data to our save_path, whether there was progress or failure.
            self.save_progress(path       = save_path,
                               album_data = retrieved_data,
                               headers    = headers)

            # Every 10 requests, take a break of 20-30 seconds.
            if (row.Index + 1) % 10 == 0:
                long_break = random.uniform(20, 30)
                logging.info("Taking a long break for %d seconds after %d request",
                             long_break, row.Index)
                await asyncio.sleep(long_break)


if __name__ == '__main__':
    CSV_PATH      = ''
    PROGRESS_PATH = ''
    FAILURE_PATH  = ''
    no_driving    = NoDriving(CSV_PATH, PROGRESS_PATH, FAILURE_PATH)

    logging.info("Starting scraping process")
    nodriver.loop().run_until_complete(no_driving.retrieve_all_albums_data())
    logging.info("Scraping process completed")
