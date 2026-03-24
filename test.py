from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL
loader = YoutubeLoaderDL.from_youtube_url(
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ", add_video_info=True
)
documents = loader.load()
print(documents[0].metadata)