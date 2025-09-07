import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# load the dataset
df = pd.read_csv("C:/Users/heath/OneDrive/Documents/Python_VS/Spotify_Project/Spotify_Youtube.csv")

# ---------- Clean data ----------
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_") #standardize (lowercase, underscores)

# FIRST,---------- SPOTIFY: collab detection + dedupe → spotify_unique ----------
sp = df.copy()

# collab detection (artist/track markers OR same (track, stream) repeating)
artist_markers = r"&|,"             # multiple artists in artist field
track_markers  = r"feat\.|ft\.|with"  # featured artist in track title

sp_text_collab = (
    sp["artist"].str.contains(artist_markers, case=False, regex=True) |
    sp["track"].str.contains(track_markers,  case=False, regex=True)
)

sp_multi_dup = sp.duplicated(subset=["track","stream"], keep=False)  # same song credited to multiple artists

sp["is_collaboration"] = sp_text_collab | sp_multi_dup

# split + dedupe (keep ALL columns)
sp_solo   = sp[~sp["is_collaboration"]].copy()
sp_collab = sp[ sp["is_collaboration"]].copy()

# solos: unique by (track, artist)
sp_solo_unique = (sp_solo.sort_values("stream", ascending=False)
                        .drop_duplicates(subset=["track","artist"], keep="first"))

# collabs: collapse to one row per track (keep highest stream)
sp_collab_unique = (sp_collab.sort_values("stream", ascending=False)
                            .drop_duplicates(subset=["track"], keep="first"))

spotify_unique = pd.concat([sp_solo_unique, sp_collab_unique], ignore_index=True)

print("Spotify rows before/after:", len(sp), "→", len(spotify_unique))

# SECOND, ---------- YOUTUBE: collab detection + dedupe → youtube_unique ----------
yt = df.copy()

yt_text_collab = (
    yt["artist"].str.contains(artist_markers, case=False, regex=True) |
    yt["track"].str.contains(track_markers,  case=False, regex=True)
)
yt_multi_dup = yt.duplicated(subset=["track","views"], keep=False)  # same video credited to multiple artists

yt["is_collaboration"] = yt_text_collab | yt_multi_dup

yt_solo   = yt[~yt["is_collaboration"]].copy()
yt_collab = yt[ yt["is_collaboration"]].copy()

yt_solo_unique = (yt_solo.sort_values("views", ascending=False)
                        .drop_duplicates(subset=["track","artist"], keep="first"))

yt_collab_unique = (yt_collab.sort_values("views", ascending=False)
                            .drop_duplicates(subset=["track"], keep="first"))

youtube_unique = pd.concat([yt_solo_unique, yt_collab_unique], ignore_index=True)

print("YouTube rows before/after:", len(yt), "→", len(youtube_unique))

# THIRD) ----------- Song Mood -------
# Function to flag songs by mood
def assign_mood(val):
    if val <= 0.33:
        return "Sad"
    elif val >= 0.66:
        return "Happy"
    else:
        return "Neutral"

# Apply to both Spotify + YouTube before combining
spotify_unique["mood"] = spotify_unique["valence"].apply(assign_mood)
youtube_unique["mood"] = youtube_unique["valence"].apply(assign_mood)

# FOURTH) ------ Engagement Ratio For Youtube ---
spotify_unique["engagement_ratio"] = (spotify_unique['likes'] + spotify_unique['comments']) / spotify_unique['views']

# FOURTH) ---------- Standardize popularity + combine ----------
# keep consistent columns
spotify_std = spotify_unique.copy()
youtube_std = youtube_unique.copy()

# Spotify
spotify_std = spotify_unique.rename(columns={"stream": "popularity"})
spotify_std["platform"] = "Spotify"
spotify_std = spotify_std[["track","artist","popularity","platform","is_collaboration","mood"]]

# YouTube
youtube_std = youtube_unique.rename(columns={"views": "popularity"})
youtube_std["platform"] = "YouTube"
youtube_std = youtube_std[["track","artist","popularity","platform","is_collaboration","mood"]]

# Combine
combined = pd.concat([spotify_std, youtube_std], ignore_index=True)

# Save
spotify_std.to_csv("spotify_clean.csv", index=False)
youtube_std.to_csv("youtube_clean.csv", index=False)
combined.to_csv("spotify_youtube_combined.csv", index=False)

combined.head()



# 1) Which songs dominate both platforms?
song_compare = (combined
                .pivot_table(index=["track","artist"], columns="platform",
                             values="popularity", aggfunc="max")
                .dropna(subset=["Spotify","YouTube"]))  # keep only songs with both

# Top by Spotify streams (with their YouTube views)
print("\nTop songs by Spotify popularity (with YouTube views):")
print(song_compare.sort_values("Spotify", ascending=False).head(10))

# Top by YouTube views (with their Spotify streams)
print("\nTop songs by YouTube popularity (with Spotify streams):")
print(song_compare.sort_values("YouTube", ascending=False).head(10))


# 2) Which artists dominate both platforms?
artist_compare = (combined.groupby(["artist","platform"])["popularity"]
                           .sum()
                           .unstack(fill_value=0)
                           .sort_values("Spotify", ascending=False))

print("\nTop artists by Spotify total popularity (with YouTube totals):")
print(artist_compare.head(10))

print("\nTop artists by YouTube total popularity (with Spotify totals):")
print(artist_compare.sort_values("YouTube", ascending=False).head(10))

# Songs big on Spotify but not YouTube
df['diff_rank'] = df['stream'].rank(ascending=False) - df['views'].rank(ascending=False)
print("Songs big on Spotify but not YouTube:")
print(df[['track','artist','stream','views','diff_rank']].sort_values('diff_rank').head(10))

# 3) Do collabs matter more on Spotify or YouTube? (Already noticed: ~24% vs ~16% of Top 100).

# 4) Correlation between Spotify popularity and YouTube views
print(df[['stream','views']].corr())

# 4.1) Is there any alignment between audio features and success across platforms?

# 5) Do sad vs happy songs perform differently across platforms? (Spotify: sad > happy; YouTube: happy > sad).



# Exporting Files
spotify_unique.to_csv("spotify_clean.csv", index=False)
youtube_unique.to_csv("youtube_clean.csv", index=False)
combined.to_csv("spotify_youtube_combined.csv", index=False)
