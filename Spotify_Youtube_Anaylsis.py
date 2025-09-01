import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# load the dataset
df = pd.read_csv("C:/Users/heath/OneDrive/Documents/Python_VS/Spotify_Project/Spotify_Youtube.csv")

#Clean data
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_") #standardize (lowercase, underscores)


# FIRST Detect collaborations
artist_markers = r"&|,"          # multiple credited artists in artist field
track_markers  = r"feat\.|ft\.|with"  # featured artists in track title

text_collab = (
    df["artist"].str.contains(artist_markers, case=False, regex=True) |
    df["track"].str.contains(track_markers,  case=False, regex=True)
)

# duplicates: same track credited to multiple artists, identical view counts
multi_artist_dup = df.duplicated(subset=["track","views"], keep=False)

# final flag for what songs are collaboration
df["is_collaboration"] = text_collab | multi_artist_dup


# SECOND, Split solos and collabs
solo   = df[~df["is_collaboration"]].copy() #False for solo
collab = df[ df["is_collaboration"]].copy()

# --- Deduplication rules ---
# Solos: keep unique by (track, artist)
solo_unique = (
    solo.sort_values("views", ascending=False)
        .drop_duplicates(subset=["track","artist"], keep="first")
)

# Collabs: collapse to one row per track (keep highest views)
collab_unique = (
    collab.sort_values("views", ascending=False)
          .drop_duplicates(subset=["track"], keep="first")
)

# THIRD Combine back
youtube_unique = pd.concat([solo_unique, collab_unique], ignore_index=True)



# 1) Which songs dominate both platforms?

# 2) Which artists dominate both platforms?


# Songs big on Spotify but not YouTube
df['diff_rank'] = df['spotify_popularity'].rank(ascending=False) - df['views'].rank(ascending=False)
df[['track_name','artist','spotify_popularity','views','diff_rank']].sort_values('diff_rank').head(10)

# 3) Do collabs matter more on Spotify or YouTube? (You already noticed: ~24% vs ~16% of Top 100).

# 4) Correlation between Spotify popularity and YouTube views
df[['stream','views']].corr()

# 4.1) Is there any alignment between audio features and success across platforms?

# 5) Do sad vs happy songs perform differently across platforms? (Spotify: sad > happy; YouTube: happy > sad).



