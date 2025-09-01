import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# load the dataset
df = pd.read_csv("C:/Users/heath/OneDrive/Documents/Python_VS/Spotify_Project/Spotify_Youtube.csv")

# Inspect structure
print(df.shape)      # rows x cols
print(df.columns)    # column names
df.head()            # preview first few rows
df.info()            # data types + null counts
df.describe()        # summary stats for numerics

#Cleaning Data
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_") #standardize (lowercase, underscores)

# Define collaboration markers
artist_markers = r"&|,"          # multiple credited artists in the artist field
track_markers  = r"feat\.|ft\.|with"  # featured artists in the track title

text_collab = (
    df["artist"].str.contains(artist_markers, case=False, regex=True) |
    df["track"].str.contains(track_markers,  case=False, regex=True)
)

# IMPORTANT: multi-artist duplicates with identical stream (same song credited to multiple artists)
# For EX: Halsey and Chainsmoker song Closer shows up as two seperate Track
multi_artist_dup = df.duplicated(subset=["track","stream"], keep=False)

# Final collab flag: text markers OR duplicate by (track, stream)
df["is_collaboration"] = text_collab | multi_artist_dup

# IMPORTANT: Hybrid deduplication
# Goal: 
# - solo songs: keep unique by (track, artist) so "Hello" by Adele vs Lionel Richie stay separate
# - collab songs: collapse to one row per track (avoid double counting across artists)

solo      = df[~df["is_collaboration"]].copy()
collab    = df[df["is_collaboration"]].copy()

# Keep ALL columns; remove dup songs
# Solos: same song title by different artists should stay separate,
# so dedupe on (track, artist)
solo_unique = (
    solo.sort_values("stream", ascending=False)
        .drop_duplicates(subset=["track", "artist"], keep="first")
)

# Collabs: same track appears under multiple artists → keep one row per track
collab_unique = (
    collab.sort_values("stream", ascending=False)
          .drop_duplicates(subset=["track"], keep="first")
)


# bring back the flag
collab_unique["is_collaboration"] = True
solo_unique["is_collaboration"]   = False

# final, deduped dataset
df_unique = pd.concat([solo_unique[["artist","track","stream","is_collaboration"]],
                       collab_unique[["artist","track","stream","is_collaboration"]]],
                      ignore_index=True)

#Spotify Analysis

#1. Top 10 most popular tracks
top_songs1 = df_unique[["track", "artist", "stream"]].dropna()
print("Spotify Top 10 Songs: ")
print(top_songs1.sort_values("stream", ascending=False).head(10))

#2. Top Artist based on the sum of the stream of all their songs
top_artist1 = df.groupby('artist')['stream'].sum().sort_values(ascending=False).head(10)
print("Spotify Top 10 Artists: ")
print(top_artist1)

#3. Top Collaborations Vs Solo Artists

#3.1) The Avg and Total
totals = df_unique.groupby("is_collaboration")["stream"].sum().rename({False:"Solo", True:"Collab"})
avgs   = df_unique.groupby("is_collaboration")["stream"].mean().rename({False:"Solo", True:"Collab"})

print("=== Total Streams (deduped) ===")
print(totals)
print("\n=== Average Streams per Song (deduped) ===")
print(avgs)

#3.2) Top lists (no double counting)
top_solo   = df_unique[~df_unique["is_collaboration"]].sort_values("stream", ascending=False).head(10)
top_collab = df_unique[df_unique["is_collaboration"]].sort_values("stream", ascending=False).head(10)

print("\n=== Top 10 Solo Songs ===")
print(top_solo[["artist","track","stream"]])

print("\n=== Top 10 Collaboration Songs ===")
print(top_collab[["artist","track","stream"]])

# 3.3) collaborations in Top 100
top100 = df_unique.sort_values("stream", ascending=False).head(100)
share_collab = (top100["is_collaboration"].mean()*100)
print(f"\n% Collaborations in Top 100 (deduped): {share_collab:.1f}%")

#4.  Correlation between audio features and popularity
# +1 = perfect positive correlation (as one goes up, the other goes up).
# -1 = perfect negative correlation (as one goes up, the other goes down).
#  0 = no correlation (no linear relationship).

audio_and_popularity_correlation = df[['stream','danceability','energy','valence','tempo']].corr()
print(" Correlation b/w  audio features and popularity \n" , audio_and_popularity_correlation)

#

#5. Happy vs Sad songs

# Combine back; this preserves ALL columns (valence, danceability, etc.)
df_unique = pd.concat([solo_unique, collab_unique], ignore_index=True)

# thresholds (tweak if dataset is skewed)
happy_threshold = 0.7
sad_threshold   = 0.3

# flag songs
df_unique['mood'] = pd.cut(
    df_unique['valence'],
    bins=[-0.01, sad_threshold, happy_threshold, 1.01],
    labels=['Sad','Neutral','Happy']
)

# 5.1) count how many songs fall into each mood
print("Song distribution by mood:")
print(df_unique['mood'].value_counts())

# 5.2) average streams by mood
avg_streams_mood = df_unique.groupby('mood')['stream'].mean().sort_values(ascending=False)
print("\nAverage Streams by Mood:")
print(avg_streams_mood)

# 5.3) total streams by mood
tot_streams_mood = df_unique.groupby('mood')['stream'].sum().sort_values(ascending=False)
print("\nTotal Streams by Mood:")
print(tot_streams_mood)

# 5.4) Top 10 Happy songs
top_happy = df_unique[df_unique['mood']=='Happy'] \
    .sort_values('stream', ascending=False).head(10)
print("\nTop 10 Happy Songs:")
print(top_happy[['artist','track','valence','stream']])

# 5.5) Top 10 Sad songs
top_sad = df_unique[df_unique['mood']=='Sad'] \
    .sort_values('stream', ascending=False).head(10)
print("\nTop 10 Sad Songs:")
print(top_sad[['artist','track','valence','stream']])

# Average mood (valence) vs energy
#df.groupby('artist')[['valence','energy']].mean().sort_values('valence', ascending=False).head(10)
