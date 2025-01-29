"""
In this file I'll be defining an Object which will facilitate the acquisition of tags such as
Genres and Descriptors from RateYourMusic.com. This will be done with the `nodriver` module which
works absolutely great. They really killed that shit, big ups to them.

NOTE Some limitations that must be taken into consideration are:
    - Suspicious behavior must be avoided to skip CAPTCHAs. This includes making too many requests
      to RYM and robotic behavior (yes, of course).
    - To avoid this, random actions must be implemented into the code to mimic human behavior. This
      includes scrolling, time.sleep() for random intervals and waiting between requests.

NOTE Additional Considerations:
    - Perhaps all songs from the same album should 
    - The Single -> Album -> EP -> ... Pipeline might be annoying. Spotify API could help
      avoid a bit of this trouble. Come back to check it (NOTE)
    - Temporary saves in the code <OR> only run it with fractions of the dataset, since running 
      the whole thing will take a long time and might run into failures.
    - For now, start off with basic code that only checks for album releases... I won't do the 
      release type pipeline yet, its annoying and will waste so many requests.
"""

import nodriver
from typing import Optional
from nodriver import Tab, Browser


class RYMScraper:
    """
    
    """

    def __init__(self):
        self.page:    Optional[Tab]     = None
        self.browser: Optional[Browser] = None

        self.rym_spans = {
                          'pri_genres' : 'release_pri_genres',
                          'sec_genres' : 'release_sec_genres',
                          'pri_desc'   : 'release_pri_descriptors'
                         }


    async def get_span_data(self):
        """_summary_
        """
        results = {}
        for span_name, span_class in self.rym_spans.items():
            element    = await self.page.select(f"span[class={span_class}]")
            all_data   = element.text_all

            temp_list = []
            for data in all_data.split(","):

                # TODO : Normalize this too later please
                curr_data = data.strip().lower()
                temp_list.append(curr_data)

            results[span_name] = temp_list


    async def get_release_data(self, artist_name : str, album_name : str, track_name : str):
        """Acquire the release data for a single track.
        
        Somewhere in this shit I gotta randomize actions to mimic human behavior
        """

        # Initialize the browser if it already hasn't been.
        if not self.browser():
            self.browser = await nodriver.start()


        # Acquire the page url and pull it up!
        url = f"https://rateyourmusic.com/release/album/{artist_name}/{album_name}/"
        self.page = await self.browser.get(url)

        # Return the track name (ID) with a dictionary which contains all of
        # tags that can be seen in `self.rym_spans`
        return {track_name : self.get_span_data()}


    async def get_several_release_data(self, track_dict : dict[str, list[str]]):
        """Acquires the release data for multiple tracks.

        Args:
            track_dict : The dictionary that contains all of the relevant data for a track.
                         this includes the artist and album name. E.g.

                            track_dict = { 'waster' : ['bladee', 'icedacer'],
                                        'scope'  : ['mssingno', 'fones']}
        """
        results = []
        for track_name, track_info in track_dict.items():
            artist_name, album_name = track_info
            release_data = await self.get_release_data(artist_name, album_name, track_name)
            
            results.append(release_data)

        return results
