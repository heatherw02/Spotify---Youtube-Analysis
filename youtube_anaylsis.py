import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# load the dataset
df = pd.read_csv("C:/Users/heath/OneDrive/Documents/Python_VS/Spotify_Project/Spotify_Youtube.csv")

# YOUTUBE ANALYSIS CLEANING 
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_") #standardize (lowercase, underscores)
#top_songs1 = df[["track", "artist", "stream"]].dropna()

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

print("Rows before:", len(df), " | after dedupe:", len(youtube_unique))
print("Columns:", youtube_unique.columns.tolist())

# Anaylsis Begins

# 1) Most Popular Songs on Youtube
print("Youtube Top 10 Songs: ")
print(youtube_unique[['track','artist','views']].sort_values("views", ascending=False).head(10))

# 2) Top Artist based on the sum of the stream of all their songs
top_artist2 = df.groupby('artist')['views'].sum().sort_values(ascending=False).head(10)
print("Youtube Top 10 Artists: ")
print(top_artist2)

# 3) SOLO VS COLLABORATION 

# 3.1) Total & average views
totals = youtube_unique.groupby("is_collaboration")["views"].sum().rename({False:"Solo", True:"Collab"})
avgs   = youtube_unique.groupby("is_collaboration")["views"].mean().rename({False:"Solo", True:"Collab"})

print("\n=== Total Views (deduped) ===")
print(totals)

print("\n=== Average Views per Video (deduped) ===")
print(avgs)

# 3.2) Top 10 solo videos
top_solo = youtube_unique[~youtube_unique["is_collaboration"]].sort_values("views", ascending=False).head(10)
print("\n=== Top 10 Solo Videos ===")
print(top_solo[["artist","track","views","likes","comments"]])

# 3.3) Top 10 collab videos
top_collab = youtube_unique[youtube_unique["is_collaboration"]].sort_values("views", ascending=False).head(10)
print("\n=== Top 10 Collaboration Videos ===")
print(top_collab[["artist","track","views","likes","comments"]])

# 3.4) % collabs in Top 100
top100 = youtube_unique.sort_values("views", ascending=False).head(100)
share_collab = (top100["is_collaboration"].mean()*100)
print(f"\n% Collaborations in Top 100 (deduped): {share_collab:.1f}%")

# 4) The Correlation
yt_corr = youtube_unique[['views','danceability','energy','valence','tempo']].corr()
print(yt_corr)

#Figure to see the correlation
plt.figure(figsize=(8,6))
sns.heatmap(yt_corr, annot=True, cmap="coolwarm")
plt.title("Correlation: YouTube Views vs Audio Features")
plt.show()

# 5) Happy Vs Sad Songs
# %% Happy vs Sad Songs on YouTube

# thresholds
happy_threshold = 0.7
sad_threshold   = 0.3

# classify songs by valence
youtube_unique['mood'] = pd.cut(
    youtube_unique['valence'],
    bins=[-0.01, sad_threshold, happy_threshold, 1.01],
    labels=['Sad','Neutral','Happy']
)

# 5.1) distribution
print("Song distribution by mood:")
print(youtube_unique['mood'].value_counts())

# 5.2) average views
avg_views_mood = youtube_unique.groupby('mood')['views'].mean().sort_values(ascending=False)
print("\nAverage Views by Mood:")
print(avg_views_mood)

# 5.3) total views
tot_views_mood = youtube_unique.groupby('mood')['views'].sum().sort_values(ascending=False)
print("\nTotal Views by Mood:")
print(tot_views_mood)

# 5.4) top 10 happy songs
top_happy = youtube_unique[youtube_unique['mood']=='Happy'] \
    .sort_values('views', ascending=False).head(10)
print("\nTop 10 Happy Songs on YouTube:")
print(top_happy[['artist','track','valence','views']])

# 5.5) top 10 sad songs
top_sad = youtube_unique[youtube_unique['mood']=='Sad'] \
    .sort_values('views', ascending=False).head(10)
print("\nTop 10 Sad Songs on YouTube:")
print(top_sad[['artist','track','valence','views']])

# 6) Engagement ratio

# Filter out videos with very low views
filtered_youtube = youtube_unique[youtube_unique['views'] > 1e5]   # keep only >100,000 views

# Recalculate top 10 engagement ratio
top_engagement = (
    filtered_youtube[['track','artist','views','likes','comments']]
    .assign(engagement_ratio=(filtered_youtube['likes'] + filtered_youtube['comments']) / filtered_youtube['views'])
    .sort_values('engagement_ratio', ascending=False)
    .head(10)
)

print("Engagement ratio:")
print(top_engagement)
