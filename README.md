# Spotify & YouTube Analysis  

## What makes a song popular?  
A song’s popularity is shaped by its **platform exposure, song characteristics (mood, audio features), and collaborations**.  
This analysis explores which factors drive **Spotify streams** and **YouTube views/engagement**.  

---

## Goal  
To answer the question: **What makes a song popular?**  

- **Platform (Spotify vs YouTube):** Do YouTube’s interactions (likes, comments, visuals) drive more engagement than Spotify streams?  
- **Song characteristics (Happy, Neutral, Sad):** Do audiences prefer upbeat songs or emotional tracks?  
- **Collaboration vs Solo:** Do collaborations between big artists outperform solo songs in popularity?  

---

## Data  
- **Source:** [Spotify–YouTube Top Music Songs Dataset](https://www.kaggle.com/code/anzarwani2/spotify-youtube-top-music-songs-eda)  
- **Size:** ~20.7k rows × 28 columns  
- **Time range:** Mixed historical snapshot (1960s–2021)  
- **Key features used:**  
  - Popularity metrics: `stream` (Spotify), `views`, `likes`, `comments` (YouTube)  
  - Audio features: `danceability`, `energy`, `valence`, `tempo`  
  - Metadata: `artist`, `track`, `platform`, `is_collaboration`, `mood`  
- **Privacy:** Public dataset, no PII (artist/song-level only).  

---

## Methods  
- **Cleaning:**  
  - Removed duplicate credits (same track listed under multiple artists).  
  - Flagged collaborations using keywords (`feat.`, `ft.`, `&`, `with`).  
  - Filtered outliers (YouTube videos with <100,000 views inflating engagement ratios).  

- **EDA:**  
  - Summaries: distribution of songs by mood, collab %, platform share.  
  - Visual checks: bar charts, scatter plots, correlation heatmaps.  

- **Analysis:**  
  - **Collab vs Solo:** Compared average streams/views, Top 10 rankings, % share in Top 100.  
  - **Mood analysis:** Classified songs into Happy, Neutral, Sad using `valence`.  
  - **Correlation:** Examined link between audio features (`danceability`, `energy`, `tempo`) and popularity.  
  - **Engagement ratio (YouTube only):** `(likes + comments) / views`.  

---

## Results (TL;DR)  
- **Collaborations dominate**: 24% of Spotify Top 100 and 16% of YouTube Top 100 are collaborations, often achieving higher average streams/views.  
- **Mood impact**: Neutral songs are most common (9.6k), but Happy songs outperform Sad songs in average views/streams.  
- **Engagement (YouTube):** Smaller artists (e.g., J-Hope, RM) achieve **10–25% engagement ratios**, far higher than global hits (<1%).  
- **Platform difference:**  
  - *Despacito* and *See You Again* dominate YouTube (8B+ views).  
  - *Blinding Lights* and *Shape of You* dominate Spotify (3B+ streams).  
- **Audio features:** Popularity weakly correlated with danceability (r ≈ 0.07) and energy (r ≈ 0.06), suggesting **song mood/artist brand matters more than raw features**.  

 *(Insert top figure)*  

---

##  Reproduce  
```bash
# Python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab

# R (optional)
# install.packages("renv"); renv::restore()


