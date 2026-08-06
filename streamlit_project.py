import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import unicodedata
from sklearn.metrics.pairwise import cosine_similarity
from mplsoccer import VerticalPitch
st.set_page_config(layout="wide")




def remove_accents(testo):
    if pd.isna(testo):
        return testo
    return ''.join(
        c for c in unicodedata.normalize('NFD', testo)
        if unicodedata.category(c) != 'Mn'
    )

def get_similarities(player_id, filename='top8_2526', num_x_cells=30, num_y_cells=30):
    sim_df = pd.read_pickle('grids/' + filename + '_30_30.pkl')
    player_names = sim_df['player_name']
    player_ids = sim_df['player_id']
    player_pd = sim_df['position_detailed']
    player_leagues = df['league']

    player_df = sim_df.loc[sim_df['player_id'] == player_id].iloc[0]
    grid = player_df['grid']
    # grid = get_grid(player_id, num_x_cells=num_x_cells, num_y_cells=num_y_cells, filename=filename)
    grid_flat = grid.flatten()
    similarities = []
    for i in sim_df.index:
        grid_compare = sim_df.loc[i]['grid']
        sim = cosine_similarity([grid_flat], [grid_compare])[0][0]
        similarities.append(sim)
    result_df = pd.DataFrame()
    result_df['player_name'] = player_names
    result_df['player_id'] = player_ids
    result_df['position'] = player_pd
    result_df['similarity'] = similarities
    result_df['league'] = player_leagues
    result_df = result_df.sort_values(by=['similarity'], ascending=False)
    return result_df

def plot_players(player_id, filename='top8_2526', num_x_cells=30, num_y_cells=30):
    sim_df = pd.read_pickle('grids/' + filename + '_30_30.pkl')
    similarities = get_similarities(player_id)
    similarities = similarities.reset_index()
    similarities = similarities.drop(columns=['index'])
    player_name1 = similarities.loc[0]['player_name']
    player_name2 = similarities.loc[1]['player_name']
    sim_2 = round(similarities.loc[1]['similarity'] * 100, 2)
    player_name3 = similarities.loc[2]['player_name']
    sim_3 = round(similarities.loc[2]['similarity'] * 100, 2)

    compare_df1 = sim_df.loc[sim_df['player_name'] == player_name1].iloc[0]
    grid_flat1 = np.array(compare_df1['grid'])
    grid1 = grid_flat1.reshape(num_x_cells,num_y_cells)

    compare_df2 = sim_df.loc[sim_df['player_name'] == player_name2].iloc[0]
    grid_flat2 = np.array(compare_df2['grid'])
    grid2 = grid_flat2.reshape(num_x_cells,num_y_cells)

    compare_df3 = sim_df.loc[sim_df['player_name'] == player_name3].iloc[0]
    grid_flat3 = np.array(compare_df3['grid'])
    grid3 = grid_flat3.reshape(num_x_cells,num_y_cells)
    # lista delle 3 matrici da plottare (sostituisci con le tue)
    matrices = [grid1, grid2, grid3]
    title2 = player_name2 + "\n(Similarity: " + str(sim_2) + "%)"
    title3 = player_name3 + "\n(Similarity: " + str(sim_3) + "%)"
    titles = [player_name1, title2, title3]  # personalizza

    pitch = VerticalPitch(pitch_type='opta', line_color='black', pitch_color='white', linewidth=1.5)

    fig, axes = plt.subplots(1, 3, figsize=(15, 9))

    x_min, x_max = pitch.dim.left, pitch.dim.right
    y_min, y_max = pitch.dim.bottom, pitch.dim.top
    x_edges = np.linspace(x_min, x_max, num_x_cells + 1)
    y_edges = np.linspace(y_min, y_max, num_y_cells + 1)

    # scala colori condivisa tra le 3 heatmap, così sono confrontabili
    vmin = min(m.min() for m in matrices)
    vmax = max(m.max() for m in matrices)

    for ax, grid, title in zip(axes, matrices, titles):
        pitch.draw(ax=ax)

        stats = {
            'statistic': grid,
            'x_grid': x_edges,
            'y_grid': y_edges,
            'cx': (x_edges[:-1] + x_edges[1:]) / 2,
            'cy': (y_edges[:-1] + y_edges[1:]) / 2,
        }

        teal_cmap = LinearSegmentedColormap.from_list(
            'teal_cmap', ['#FFFFFF', '#00695c']  # da teal chiarissimo a teal scuro
        )
        pcm = pitch.heatmap(stats, ax=ax, cmap=teal_cmap, edgecolors='none',
                            alpha=0.75, zorder=1, vmin=vmin, vmax=vmax)
        pitch.draw(ax=ax)  # ridisegna le linee sopra
        ax.set_title(title, fontsize=12)

    fig.colorbar(pcm, ax=axes, shrink=0.6, orientation='horizontal', pad=0.05)
    # plt.show()
    return fig, similarities


df = pd.read_pickle('grids/top8_2526_30_30.pkl')
df = df.sort_values(by=['player_name'])
df["player_name"] = df["player_name"].apply(remove_accents)



st.title("Heatmaps Similarity")
st.subheader("Search for a player in order to find the most similar players!")
st.write("Last Update: August 6th, 2026")

st.info("This project runs a similarity algorithm, based on player heatmaps. Note therefore that the similarity is based only on movement.  \nData are taken from the 2025/26 season of the top 8 European Leagues (England, Spain, Italy, Germany, France, Netherlands, Belgium, Portugal).")

# st.write(df)
player_name = st.selectbox(
    "Search for a Player",
    options=df['player_name'],
    index=None,
    placeholder="Type a Player Name"
)

if player_name:
    row = df.loc[df['player_name'] == player_name].iloc[0]
    player_id = row['player_id']
    fig, similarities = plot_players(player_id=player_id)
    st.pyplot(fig)

    top10 = similarities.iloc[1:11].reset_index(drop=True)
    st.subheader("Similarity Top 10")

    col_left, col_right = st.columns(2)

    with col_left:
        for i, row in top10.iloc[0:5].iterrows():
            with st.container(border=True):
                st.markdown(f"**#{i+1} — {row['player_name']}**")
                st.progress(row['similarity'])
                st.caption(f"{row['similarity']:.1%}")

    with col_right:
        for i, row in top10.iloc[5:10].iterrows():
            with st.container(border=True):
                st.markdown(f"**#{i+1} — {row['player_name']}**")
                st.progress(row['similarity'])
                st.caption(f"{row['similarity']:.1%}")