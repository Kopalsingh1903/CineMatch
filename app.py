import streamlit as st
import pandas as pd
import numpy as np
import json
import ast
import altair as alt

st.set_page_config(page_title="CineMatch", layout="wide", initial_sidebar_state="expanded")

# ---- Netflix CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    background-color: #141414 !important;
    color: #e5e5e5 !important;
    font-family: 'Inter', sans-serif !important;
}
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding: 2rem 3rem !important;}

section[data-testid="stSidebar"] {
    background-color: #000000 !important;
    border-right: 1px solid #2a2a2a;
}
section[data-testid="stSidebar"] * {color: #e5e5e5 !important;}

.netflix-logo {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 72px !important;
    color: #E50914 !important;
    letter-spacing: 4px;
    line-height: 1;
    margin-bottom: 0;
}
.netflix-tagline {
    font-size: 16px;
    color: #808080;
    letter-spacing: 1px;
    margin-top: 4px;
    margin-bottom: 32px;
}
.movie-card {
    background: #1f1f1f;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 20px;
    margin-bottom: 16px;
}
.movie-card:hover {border-color: #E50914;}
.movie-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 28px;
    color: #ffffff;
    letter-spacing: 1.5px;
    margin: 0 0 4px 0;
}
.movie-year {font-size: 14px; color: #808080; margin-bottom: 12px;}
.match-badge {font-family: 'Bebas Neue', sans-serif; font-size: 36px; color: #46d369; letter-spacing: 2px;}
.match-label {font-size: 11px; color: #808080; letter-spacing: 2px; text-transform: uppercase;}
.rating-badge {font-family: 'Bebas Neue', sans-serif; font-size: 28px; color: #E50914; letter-spacing: 1px;}
.meta-label {font-size: 11px; color: #808080; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 2px;}
.meta-value {font-size: 14px; color: #e5e5e5; margin-bottom: 10px;}
.genre-tag {
    display: inline-block; background: #2a2a2a; color: #e5e5e5;
    font-size: 11px; padding: 3px 10px; border-radius: 20px;
    margin: 2px; border: 1px solid #3a3a3a;
}
.cine-link {
    display: inline-block; font-size: 13px; font-weight: 600;
    color: #ffffff; background: #E50914; padding: 6px 14px;
    border-radius: 4px; text-decoration: none; margin-right: 8px; margin-top: 10px;
}
.cine-link-ghost {
    display: inline-block; font-size: 13px; font-weight: 600;
    color: #e5e5e5; background: transparent; border: 1px solid #808080;
    padding: 6px 14px; border-radius: 4px; text-decoration: none; margin-top: 10px;
}
.section-heading {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 24px; color: #e5e5e5;
    letter-spacing: 2px; margin: 24px 0 12px 0;
}
.watched-tag {
    font-size: 11px; color: #46d369; border: 1px solid #46d369;
    padding: 2px 8px; border-radius: 4px; display: inline-block; margin-top: 6px;
}
.insight-card {
    background: #1f1f1f; border: 1px solid #2a2a2a;
    border-radius: 6px; padding: 20px; margin-bottom: 16px;
}
.stButton > button {
    background-color: #E50914 !important;
    color: white !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 10px 24px !important;
    font-size: 15px !important;
}
.stButton > button:hover {background-color: #f40612 !important;}
hr {border-color: #2a2a2a !important;}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background-color: #000000;
    border-bottom: 1px solid #2a2a2a;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 16px !important;
    letter-spacing: 2px !important;
    color: #808080 !important;
    padding: 12px 24px !important;
}
.stTabs [aria-selected="true"] {
    color: #E50914 !important;
    border-bottom: 2px solid #E50914 !important;
}
</style>
""", unsafe_allow_html=True)

# ---- Session state ----
for key, default in [('watched', []), ('results', []), ('selected_titles', []), ('cold_results', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---- Load data ----
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_movies_enriched.csv")
    similarity_matrix = np.load("similarity_matrix.npy")
    with open("title_to_index.json", "r", encoding="utf-8") as f:
        title_to_index = json.load(f)
    return df, similarity_matrix, title_to_index

df, similarity_matrix, title_to_index = load_data()

# ---- Helpers ----
def parse_list_field(value):
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return ', '.join(parsed)
    except:
        pass
    return str(value)

ALL_GENRES = sorted(set(
    g.strip() for genres in df['genres'].dropna()
    for g in str(genres).split(',')
    if g.strip()
))

def get_recommendations(selected_titles, watched_titles, genre_filter=None, year_range=None, n=5):
    seed_indices = [title_to_index[t] for t in selected_titles if t in title_to_index]
    if not seed_indices:
        return []
    avg_scores = np.mean([similarity_matrix[i] for i in seed_indices], axis=0)
    sim_scores = sorted(enumerate(avg_scores), key=lambda x: x[1], reverse=True)
    exclude = [t.lower() for t in selected_titles + watched_titles]
    results = []
    for idx, score in sim_scores:
        movie = df.iloc[idx]
        if movie['title'].lower() in exclude:
            continue
        if genre_filter and genre_filter != 'All':
            if genre_filter not in str(movie['genres']):
                continue
        if year_range:
            year = movie['release_year']
            if pd.isna(year) or not (year_range[0] <= year <= year_range[1]):
                continue
        results.append((idx, score))
        if len(results) == n:
            break
    return results

# ---- Feature 1: Taste Profile Chart (Altair) ----
def build_radar(selected_titles):
    genre_counts = {g: 0 for g in ALL_GENRES}
    for title in selected_titles:
        matches = df[df['title'] == title]
        if not matches.empty:
            genres = str(matches.iloc[0]['genres']).split(',')
            for g in genres:
                g = g.strip()
                if g in genre_counts:
                    genre_counts[g] += 1

    top_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)[:10]
    radar_df = pd.DataFrame({
        'Genre': top_genres,
        'Score': [genre_counts[g] for g in top_genres]
    })

    chart = alt.Chart(radar_df).mark_bar().encode(
        x=alt.X('Score:Q', axis=alt.Axis(labelColor='#e5e5e5', titleColor='#e5e5e5', gridColor='#2a2a2a')),
        y=alt.Y('Genre:N', sort='-x', axis=alt.Axis(labelColor='#e5e5e5', titleColor='#e5e5e5')),
        color=alt.value('#E50914')
    ).properties(
        background='#141414',
        height=300
    ).configure_view(strokeOpacity=0)
    return chart

# ---- Feature 6: Cold Start recommendations by genre ----
def cold_start_recommend(liked_genres, year_range=None, n=5):
    results = []
    df_filtered = df.copy()
    if year_range:
        df_filtered = df_filtered[
            (df_filtered['release_year'] >= year_range[0]) &
            (df_filtered['release_year'] <= year_range[1])
        ]
    for idx, row in df_filtered.iterrows():
        movie_genres = [g.strip() for g in str(row['genres']).split(',')]
        overlap = len(set(liked_genres) & set(movie_genres))
        if overlap > 0:
            score = overlap / len(liked_genres)
            results.append((idx, score))
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results[:n]

# ---- Sidebar ----
with st.sidebar:
    st.markdown('<div class="netflix-logo">CINEMATCH</div>', unsafe_allow_html=True)
    st.markdown('<div class="netflix-tagline">PERSONALISED PICKS · POWERED BY ML</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="section-heading">FILTERS</div>', unsafe_allow_html=True)
    genre_filter = st.selectbox("Genre", ['All'] + ALL_GENRES)
    min_year = int(df['release_year'].min())
    max_year = int(df['release_year'].max())
    year_range = st.slider("Release Year", min_year, max_year, (2000, max_year))

    st.divider()

    if st.session_state.watched:
        st.markdown('<div class="section-heading">ALREADY WATCHED</div>', unsafe_allow_html=True)
        for w in st.session_state.watched:
            st.markdown(f'<div class="watched-tag">✓ {w}</div>', unsafe_allow_html=True)
        st.write("")
        if st.button("Clear List"):
            st.session_state.watched = []
            st.session_state.results = []
            st.rerun()

    st.divider()
    st.caption(f"{len(df)} movies · Content-based filtering · TMDb data")

# ---- Main header ----
st.markdown('<div class="netflix-logo">CINEMATCH</div>', unsafe_allow_html=True)
st.markdown('<div class="netflix-tagline">TELL US WHAT YOU\'VE WATCHED. WE\'LL FIND WHAT\'S NEXT.</div>', unsafe_allow_html=True)

# ---- TABS ----
tab1, tab2, tab3 = st.tabs(["🎬  RECOMMENDER", "📊  DATASET INSIGHTS", "🌱  NEW HERE?"])


# TAB 1 — Main recommender

with tab1:
    st.divider()
    st.markdown('<div class="section-heading">WHAT HAVE YOU WATCHED?</div>', unsafe_allow_html=True)

    all_titles = sorted(df['title'].tolist())
    selected_titles = st.multiselect(
        "Pick 2–3 movies you've already seen:",
        options=all_titles,
        max_selections=3
    )

    if selected_titles:
        st.markdown('<div class="section-heading">YOUR TASTE PROFILE</div>', unsafe_allow_html=True)
        st.caption("Built from the genres of your selected movies — your cinematic fingerprint.")
        fig_radar = build_radar(selected_titles)
        st.altair_chart(fig_radar, use_container_width=True)

    if len(selected_titles) < 2:
        st.info("Select at least 2 movies to get recommendations.")
        st.stop()

    if st.button("▶  FIND MY NEXT WATCH"):
        st.session_state.selected_titles = selected_titles
        st.session_state.results = get_recommendations(
            selected_titles,
            st.session_state.watched,
            genre_filter=genre_filter,
            year_range=year_range
        )

    if st.session_state.results:
        st.divider()
        st.markdown('<div class="section-heading">RECOMMENDED FOR YOU</div>', unsafe_allow_html=True)

        if not st.session_state.results:
            st.warning("No results — try adjusting your filters.")

        for i, (idx, score) in enumerate(st.session_state.results):
            movie = df.iloc[idx]
            director = parse_list_field(movie['director'])
            cast = parse_list_field(movie['cast'])
            genres = [g.strip() for g in str(movie['genres']).split(',')]
            genre_tags = ' '.join([f'<span class="genre-tag">{g}</span>' for g in genres])
            clean_title = movie['title'].strip().replace(' ', '+')
            trailer_url = f"https://www.youtube.com/results?search_query={clean_title}+{int(movie['release_year'])}+official+trailer"
            justwatch_url = f"https://www.justwatch.com/in/search?q={clean_title}"
            match_pct = min(int(score * 100), 99)

            col_poster, col_info, col_meta = st.columns([1, 3, 1])

            with col_poster:
                if pd.notna(movie.get("poster_path")):
                    st.image("https://image.tmdb.org/t/p/w500" + str(movie["poster_path"]), use_container_width=True)
                else:
                    st.markdown("""<div style='background:#2a2a2a;height:200px;border-radius:4px;
                        display:flex;align-items:center;justify-content:center;color:#555;font-size:32px;'>🎬</div>""",
                        unsafe_allow_html=True)

            with col_info:
                st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">{movie['title']}</div>
                        <div class="movie-year">{int(movie['release_year'])}</div>
                        <div class="meta-label">Director</div>
                        <div class="meta-value">{director}</div>
                        <div class="meta-label">Cast</div>
                        <div class="meta-value">{cast}</div>
                        <div class="meta-label">Genres</div>
                        <div style="margin-bottom:12px">{genre_tags}</div>
                        <a href="{trailer_url}" target="_blank" class="cine-link">▶ Trailer</a>
                        <a href="{justwatch_url}" target="_blank" class="cine-link-ghost">📺 Where to Watch</a>
                    </div>
                """, unsafe_allow_html=True)

            with col_meta:
                st.markdown(f"""
                    <div style="text-align:center;padding:16px 0;">
                        <div class="match-label">Match</div>
                        <div class="match-badge">{match_pct}%</div>
                        <br>
                        <div class="match-label">Rating</div>
                        <div class="rating-badge">{movie['vote_average']:.1f} ★</div>
                    </div>
                """, unsafe_allow_html=True)

                if st.button("✓ Watched", key=f"watched_{i}_{idx}"):
                    st.session_state.watched.append(movie['title'])
                    st.session_state.results = get_recommendations(
                        st.session_state.selected_titles,
                        st.session_state.watched,
                        genre_filter=genre_filter,
                        year_range=year_range
                    )
                    st.rerun()

            st.divider()


# TAB 2 — Dataset Insights

with tab2:
    st.divider()
    st.markdown('<div class="section-heading">ABOUT THE DATA</div>', unsafe_allow_html=True)
    st.caption(f"Built from {len(df)} movies pulled via TMDb API · Filtered to ≥20 votes for rating reliability")

    col_a, col_b = st.columns(2)

    # Chart 1: Genre distribution
    with col_a:
        st.markdown('<div class="section-heading">GENRE DISTRIBUTION</div>', unsafe_allow_html=True)
        genre_counts_df = (
            df['genres'].dropna()
            .str.split(',').explode()
            .str.strip().value_counts()
            .head(15)
            .reset_index()
        )
        genre_counts_df.columns = ['Genre', 'Count']
        fig1 = alt.Chart(genre_counts_df).mark_bar().encode(
            x=alt.X('Count:Q', axis=alt.Axis(labelColor='#e5e5e5', titleColor='#e5e5e5', gridColor='#2a2a2a')),
            y=alt.Y('Genre:N', sort='-x', axis=alt.Axis(labelColor='#e5e5e5', titleColor='#e5e5e5')),
            color=alt.value('#E50914')
        ).properties(
            background='#141414',
            height=380
        ).configure_view(strokeOpacity=0)
        st.altair_chart(fig1, use_container_width=True)

    # Chart 2: Rating distribution
    with col_b:
        st.markdown('<div class="section-heading">RATING DISTRIBUTION</div>', unsafe_allow_html=True)
        fig2 = alt.Chart(df).mark_bar(color='#E50914').encode(
            x=alt.X('vote_average:Q', bin=alt.Bin(maxbins=25),
                    axis=alt.Axis(labelColor='#e5e5e5', titleColor='#e5e5e5', gridColor='#2a2a2a'),
                    title='Rating'),
            y=alt.Y('count():Q',
                    axis=alt.Axis(labelColor='#e5e5e5', titleColor='#e5e5e5', gridColor='#2a2a2a'),
                    title='Number of Movies')
        ).properties(
            background='#141414',
            height=380
        ).configure_view(strokeOpacity=0)
        st.altair_chart(fig2, use_container_width=True)

    st.divider()
    col_c, col_d = st.columns(2)

    # Chart 3: Popularity vs Rating scatter
    with col_c:
        st.markdown('<div class="section-heading">POPULARITY VS RATING</div>', unsafe_allow_html=True)
        st.caption("Correlation: ~0.05 — essentially independent signals")
        fig3 = alt.Chart(df).mark_circle(color='#E50914', opacity=0.5).encode(
            x=alt.X('vote_average:Q', title='Rating',
                    axis=alt.Axis(labelColor='#e5e5e5', titleColor='#e5e5e5', gridColor='#2a2a2a')),
            y=alt.Y('popularity:Q', title='Popularity',
                    axis=alt.Axis(labelColor='#e5e5e5', titleColor='#e5e5e5', gridColor='#2a2a2a')),
            tooltip=['title', 'vote_average', 'popularity']
        ).properties(
            background='#141414',
            height=380
        ).configure_view(strokeOpacity=0)
        st.altair_chart(fig3, use_container_width=True)

    # Chart 4: Genre trend over time
    with col_d:
        st.markdown('<div class="section-heading">GENRE SHARE OVER TIME</div>', unsafe_allow_html=True)
        st.caption("Normalized % share per year — Action declining, Thriller rising")
        genre_year = df[['release_year', 'genres']].dropna().copy()
        genre_year['genres'] = genre_year['genres'].str.split(',')
        genre_year = genre_year.explode('genres')
        genre_year['genres'] = genre_year['genres'].str.strip()
        pivot = genre_year.groupby(['release_year', 'genres']).size().unstack(fill_value=0)
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        pivot_pct = pivot_pct[pivot_pct.index >= 2017]

        trend_df = pivot_pct[['Action', 'Horror', 'Thriller']].reset_index().melt(
            id_vars='release_year',
            var_name='Genre',
            value_name='Share'
        )

        color_scale = alt.Scale(
            domain=['Action', 'Horror', 'Thriller'],
            range=['#808080', '#E50914', '#46d369']
        )

        fig4 = alt.Chart(trend_df).mark_line(point=True).encode(
            x=alt.X('release_year:O', title='Year',
                    axis=alt.Axis(labelColor='#e5e5e5', titleColor='#e5e5e5', gridColor='#2a2a2a')),
            y=alt.Y('Share:Q', title='% Share',
                    axis=alt.Axis(labelColor='#e5e5e5', titleColor='#e5e5e5', gridColor='#2a2a2a')),
            color=alt.Color('Genre:N', scale=color_scale,
                            legend=alt.Legend(labelColor='#e5e5e5', titleColor='#e5e5e5')),
            strokeWidth=alt.value(2)
        ).properties(
            background='#141414',
            height=380
        ).configure_view(strokeOpacity=0)
        st.altair_chart(fig4, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-heading">TOP 10 HIGHEST RATED MOVIES</div>', unsafe_allow_html=True)
    top10 = df[df['vote_count'] >= 20].sort_values('vote_average', ascending=False).head(10)
    top10_display = top10[['title', 'genres', 'vote_average', 'vote_count', 'release_year']].copy()
    top10_display.columns = ['Title', 'Genres', 'Rating', 'Votes', 'Year']
    st.dataframe(top10_display.reset_index(drop=True), use_container_width=True, hide_index=True)


# TAB 3 — Cold Start (New User Mode)

with tab3:
    st.divider()
    st.markdown('<div class="section-heading">NEW HERE? NO WATCH HISTORY?</div>', unsafe_allow_html=True)
    st.write("No problem. Tell us which **genres** you enjoy — we'll find great movies for you without needing your viewing history.")
    st.caption("This solves what data scientists call the 'cold start problem' — recommending to someone with no history.")

    st.divider()

    liked_genres = st.multiselect(
        "Pick the genres you enjoy:",
        options=ALL_GENRES,
        default=None,
        placeholder="Choose genres..."
    )

    cold_year_range = st.slider(
        "Preferred era:",
        min_year, max_year, (2010, max_year),
        key="cold_year"
    )

    if not liked_genres:
        st.info("Pick at least one genre to get started.")
        st.stop()

    if st.button("▶  FIND MOVIES FOR ME", key="cold_btn"):
        st.session_state.cold_results = cold_start_recommend(
            liked_genres,
            year_range=cold_year_range
        )

    if st.session_state.cold_results:
        st.divider()
        st.markdown('<div class="section-heading">PICKED FOR YOU</div>', unsafe_allow_html=True)

        for i, (idx, score) in enumerate(st.session_state.cold_results):
            movie = df.iloc[idx]
            director = parse_list_field(movie['director'])
            cast = parse_list_field(movie['cast'])
            genres = [g.strip() for g in str(movie['genres']).split(',')]
            genre_tags = ' '.join([f'<span class="genre-tag">{g}</span>' for g in genres])
            clean_title = movie['title'].strip().replace(' ', '+')
            trailer_url = f"https://www.youtube.com/results?search_query={clean_title}+{int(movie['release_year'])}+official+trailer"
            justwatch_url = f"https://www.justwatch.com/in/search?q={clean_title}"

            col_poster, col_info, col_meta = st.columns([1, 3, 1])

            with col_poster:
                if pd.notna(movie.get("poster_path")):
                    st.image("https://image.tmdb.org/t/p/w500" + str(movie["poster_path"]), use_container_width=True)
                else:
                    st.markdown("""<div style='background:#2a2a2a;height:200px;border-radius:4px;
                        display:flex;align-items:center;justify-content:center;color:#555;font-size:32px;'>🎬</div>""",
                        unsafe_allow_html=True)

            with col_info:
                genre_overlap = set(liked_genres) & set(genres)
                st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">{movie['title']}</div>
                        <div class="movie-year">{int(movie['release_year'])}</div>
                        <div class="meta-label">Director</div>
                        <div class="meta-value">{director}</div>
                        <div class="meta-label">Cast</div>
                        <div class="meta-value">{cast}</div>
                        <div class="meta-label">Genres</div>
                        <div style="margin-bottom:8px">{genre_tags}</div>
                        <div class="meta-label">Why this</div>
                        <div class="meta-value">Matches your interest in: {', '.join(genre_overlap)}</div>
                        <a href="{trailer_url}" target="_blank" class="cine-link">▶ Trailer</a>
                        <a href="{justwatch_url}" target="_blank" class="cine-link-ghost">📺 Where to Watch</a>
                    </div>
                """, unsafe_allow_html=True)

            with col_meta:
                genre_match_pct = min(int(score * 100), 99)
                st.markdown(f"""
                    <div style="text-align:center;padding:16px 0;">
                        <div class="match-label">Genre Match</div>
                        <div class="match-badge">{genre_match_pct}%</div>
                        <br>
                        <div class="match-label">Rating</div>
                        <div class="rating-badge">{movie['vote_average']:.1f} ★</div>
                    </div>
                """, unsafe_allow_html=True)

            st.divider()
