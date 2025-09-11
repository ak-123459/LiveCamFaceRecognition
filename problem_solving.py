from youtubesearchpython import VideosSearch


def search_youtube(query):

    videosSearch = VideosSearch(query, limit=1)
    result = videosSearch.result()

    if result['result']:
        video_url = result['result'][0]['link']
        print(f"Top result: {video_url}")
        return video_url
    else:
        print("No results found.")
        return None


# Example usage
search_youtube("Shape of You Ed Sheeran")
