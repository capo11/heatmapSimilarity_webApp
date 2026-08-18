import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import unicodedata
from unidecode import unidecode
from sklearn.metrics.pairwise import cosine_similarity
from mplsoccer import VerticalPitch
from streamlit_option_menu import option_menu
from matplotlib.colors import Normalize
from matplotlib.colors import PowerNorm

played_matches_threshold = 10

st.set_page_config(layout="wide")

def find_alt_name(player_name):
    df = pd.read_csv('utility/alt_names.csv')
    player_df = df.loc[df['name'] == player_name]
    if len(player_df) == 0:
        player_df = df.loc[df['alt_name'] == player_name]
        if len(player_df) == 0:
            st.error(f"Alternative name for player {player_name} not found!")
        else:
            row = player_df.iloc[0]
            alt_name = row['name']
    else:
        row = player_df.iloc[0]
        alt_name = row['alt_name']
    return alt_name

def top_percs(grid, num_x_cells, num_y_cells, top):
    vals = []
    for idx, perc in np.ndenumerate(grid):
        state_start, state_end = idx
        x_start, y_start = divmod(state_start, num_x_cells)
        x_end, y_end = divmod(state_end, num_x_cells)
        if not (x_start == x_end and y_start == y_end) and perc > 0:
            vals.append(perc)
    return np.sort(np.array(vals))[::-1][:top]

def get_similarities(player_id, filename='top8_2526', num_x_cells=30, num_y_cells=30):
    sim_df = pd.read_pickle('grids/' + filename + '_30_30.pkl')
    sim_df = sim_df.loc[sim_df['played_matches'] >= played_matches_threshold]
    player_names = sim_df['player_name']
    player_ids = sim_df['player_id']
    player_pd = sim_df['position_detailed']
    if 'league' in sim_df.columns:
        player_leagues = sim_df['league']
    if 'team' in sim_df.columns:
        player_teams = sim_df['team']

    player_df = sim_df.loc[sim_df['player_id'] == player_id]
    if len(player_df)>1:
        st.error("Player has more than one team")
    else:
        player_df = player_df.iloc[0]
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
    if 'league' in sim_df.columns:
        result_df['league'] = player_leagues
    if 'team' in sim_df.columns:
        result_df['team'] = player_teams
    result_df = result_df.sort_values(by=['similarity'], ascending=False)
    return result_df

def plot_players(player_id, filename='top8_2526', num_x_cells=30, num_y_cells=30):
    sim_df = pd.read_pickle('grids/' + filename + '_30_30.pkl')
    sim_df = sim_df.loc[sim_df['played_matches'] >= played_matches_threshold]
    similarities = get_similarities(player_id)
    similarities = similarities.reset_index()
    similarities = similarities.drop(columns=['index'])
    player_name1 = similarities.loc[0]['player_name']
    league1 = similarities.loc[0]['league']
    team1 = similarities.loc[0]['team']
    player_name2 = similarities.loc[1]['player_name']
    league2 = similarities.loc[1]['league']
    team2 = similarities.loc[1]['team']
    sim_2 = round(similarities.loc[1]['similarity'] * 100, 2)
    player_name3 = similarities.loc[2]['player_name']
    league3 = similarities.loc[2]['league']
    team3 = similarities.loc[2]['team']
    sim_3 = round(similarities.loc[2]['similarity'] * 100, 2)

    compare_df1 = sim_df.loc[sim_df['player_name'].str.lower() == player_name1.lower()].iloc[0]
    grid_flat1 = np.array(compare_df1['grid'])
    grid1 = grid_flat1.reshape(num_x_cells,num_y_cells)

    compare_df2 = sim_df.loc[sim_df['player_name'].str.lower() == player_name2.lower()].iloc[0]
    grid_flat2 = np.array(compare_df2['grid'])
    grid2 = grid_flat2.reshape(num_x_cells,num_y_cells)

    compare_df3 = sim_df.loc[sim_df['player_name'].str.lower() == player_name3.lower()].iloc[0]
    grid_flat3 = np.array(compare_df3['grid'])
    grid3 = grid_flat3.reshape(num_x_cells,num_y_cells)
    # lista delle 3 matrici da plottare (sostituisci con le tue)
    matrices = [grid1, grid2, grid3]
    title1 = player_name1 + " - " + team1
    title2 = player_name2 + " - " + team2 + "\n(Similarity: " + str(sim_2) + "%)"
    title3 = player_name3 + " - " + team3 + "\n(Similarity: " + str(sim_3) + "%)"
    titles = [title1, title2, title3]  # personalizza

    # pitch = VerticalPitch(pitch_type='opta', line_color='black', pitch_color='white', linewidth=1.5)
    pitch = VerticalPitch(pitch_type='opta', line_color='white', pitch_color='none', linewidth=1.5)

    fig, axes = plt.subplots(1, 3, figsize=(15, 9))

    fig.patch.set_alpha(0)
    for ax in axes:
        ax.patch.set_alpha(0)

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


        dark_blue_red_cmap = LinearSegmentedColormap.from_list(
            'dark_blue_red', ['#0e1117', '#e63946']
        )
        teal_cmap = LinearSegmentedColormap.from_list(
            'teal_cmap', ['#FFFFFF', '#0000FF']  # da teal chiarissimo a teal scuro
        )
        pcm = pitch.heatmap(stats, ax=ax, cmap=dark_blue_red_cmap, edgecolors='none',
                            alpha=0.75, zorder=1, vmin=vmin, vmax=vmax)
        pitch.draw(ax=ax)  # ridisegna le linee sopra
        ax.set_title(title, fontsize=12, color='white')
        

    # fig.colorbar(pcm, ax=axes, shrink=0.6, orientation='horizontal', pad=0.05)

    cbar = fig.colorbar(pcm, ax=axes, shrink=0.6, orientation='horizontal', pad=0.05)
    cbar.ax.xaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')

    # plt.show()
    return fig, similarities

def plot_players_movements(player_id, filename='seriea_2526', num_x_cells=5, num_y_cells=5, top=20):
    # touches_df = pd.read_csv('touches/' + filename + '.csv')
    # touches_df = touches_df.dropna(subset=['endX', 'endY'])
    similarities = get_similarities_movements(player_id=player_id, filename=filename, num_x_cells=num_x_cells, num_y_cells=num_y_cells).reset_index()
    similarities = similarities.drop(columns=['index'])
    # st.write(similarities)

    grid_shape = num_x_cells*num_y_cells
    player_name1 = similarities.loc[0]['player_name']
    grid1 = similarities.loc[0]['grid'].reshape((grid_shape,grid_shape))
    league1 = similarities.loc[0]['league']
    team1 = similarities.loc[0]['team']
    player_name2 = similarities.loc[1]['player_name']
    grid2 = similarities.loc[1]['grid'].reshape((grid_shape,grid_shape))
    league2 = similarities.loc[1]['league']
    team2 = similarities.loc[1]['team']
    sim_2 = round(similarities.loc[1]['similarity'] * 100, 2)
    player_name3 = similarities.loc[2]['player_name']
    grid3 = similarities.loc[2]['grid'].reshape((grid_shape,grid_shape))
    league3 = similarities.loc[2]['league']
    team3 = similarities.loc[2]['team']
    sim_3 = round(similarities.loc[2]['similarity'] * 100, 2)


    all_top_vals = np.concatenate([
        top_percs(grid1, num_x_cells, num_y_cells, top),
        top_percs(grid2, num_x_cells, num_y_cells, top),
        top_percs(grid3, num_x_cells, num_y_cells, top),
    ])
    vmin, vmax = all_top_vals.min(), all_top_vals.max()

    # st.write(grid1)
    # st.write(grid1.shape)

    pitch = VerticalPitch(pitch_type='opta', line_color='white', pitch_color='none', linewidth=1.5)
    fig, axes = plt.subplots(1, 3, figsize=(15, 9))
    fig.patch.set_alpha(0)
    for ax in axes:
        pitch.draw(ax=ax)
        ax.patch.set_alpha(0)
    axes[0].set_title(f"{player_name1} - {team1}", color='white')
    axes[1].set_title(f"{player_name2} - {team2}\n(Similarity: {str(sim_2)}%)", color='white')
    axes[2].set_title(f"{player_name3} - {team3}\n(Similarity: {str(sim_3)}%)", color='white')
    fig, pcm1 = plot_movements(grid1, fig, axes[0], pitch, num_x_cells, num_y_cells, vmin=vmin, vmax=vmax, top=top)
    fig, pcm2 = plot_movements(grid2, fig, axes[1], pitch, num_x_cells, num_y_cells, vmin=vmin, vmax=vmax, top=top)
    fig, pcm3 = plot_movements(grid3, fig, axes[2], pitch, num_x_cells, num_y_cells, vmin=vmin, vmax=vmax, top=top)

    cbar = fig.colorbar(pcm1, ax=axes, shrink=0.6, orientation='horizontal', pad=0.05)
    cbar.ax.xaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')
    cbar.outline.set_edgecolor('white')
    
    st.pyplot(fig)
    

    top10 = similarities.iloc[1:11].reset_index(drop=True)
    st.subheader(f"Similarity Top 10 - {player_name1}")

    col_left, col_right = st.columns(2)

    with col_left:
        for i, row in top10.iloc[0:5].iterrows():
            with st.container(border=True):
                st.markdown(f"**#{i+1} — {row['player_name']} - {row['team']}**")
                # st.markdown(f"**#{i+1} — {row['player_name']} - {row['team']} - {row['position']}**")
                st.progress(row['similarity'])
                st.caption(f"**{row['similarity']:.1%}**")
                with st.expander("Compare players"):
                    # st.write(player_name1, row['player_name'])
                    compare_players(player_name1, row['player_name'], filename=filename, num_x_cells_tou=num_x_cells, num_y_cells_tou=num_y_cells, top=top, expander=True, type='movement')

    with col_right:
        for i, row in top10.iloc[5:10].iterrows():
            with st.container(border=True):
                st.markdown(f"**#{i+1} — {row['player_name']} - {row['team']}**")
                # st.markdown(f"**#{i+1} — {row['player_name']} - {row['team']} - {row['position']}**")
                st.progress(row['similarity'])
                st.caption(f"**{row['similarity']:.1%}**")
                with st.expander("Compare players"):
                    compare_players(player_name1, row['player_name'], filename=filename, num_x_cells_tou=num_x_cells, num_y_cells_tou=num_y_cells, top=top, expander=True, type='movement')
                

def plot_movements(grid, fig, ax, pitch, num_x_cells=5, num_y_cells=5, vmin=None, vmax=None, top=20):
    xs = []
    ys = []
    xe = []
    ye = []
    percs = []
    for idx, perc in np.ndenumerate(grid):
        state_start = idx[0]
        state_end = idx[1]
        x_start, y_start = divmod(state_start, num_x_cells)
        x_end, y_end = divmod(state_end, num_x_cells)
        if not (x_start==x_end and y_start==y_end):
            xs.append(x_start)
            ys.append(y_start)
            xe.append(x_end)
            ye.append(y_end)
            percs.append(perc)
        # print(state_start, state_end, (x_start, y_start), (x_end, y_end), perc)

    new_df = pd.DataFrame()
    new_df['x_start'] = xs
    new_df['y_start'] = ys
    new_df['x_end'] = xe
    new_df['y_end'] = ye
    new_df['perc'] = percs
    new_df = new_df.loc[new_df['perc']>0]
    new_df = new_df.sort_values(by=['perc'], ascending=False)
    new_df = new_df.head(top)
    # st.write(new_df)

    teal_cmap = LinearSegmentedColormap.from_list(
                'teal_cmap', ['#FFFFFF', '#00695c']  # da teal chiarissimo a teal scuro
            )

    cell_length = 100/num_x_cells
    cell_offset = cell_length/2

    # norm = Normalize(vmin=vmin, vmax=vmax)
    norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)

    x_lines = np.linspace(0, 100, num_x_cells+1)[1:-1]
    y_lines = np.linspace(0, 100, num_y_cells+1)[1:-1]
    pitch.lines(x_lines, np.zeros_like(x_lines),
                x_lines, np.full_like(x_lines, 100),
                ax=ax, lw=1, linestyle='--', color='white', alpha=0.3)
    pitch.lines(np.zeros_like(y_lines), y_lines,
                    np.full_like(y_lines, 100), y_lines,
                    ax=ax, lw=1, linestyle='--', color='white', alpha=0.3)


    pcm = pitch.arrows((new_df['x_start']*cell_length)+cell_offset, (new_df['y_start']*cell_length)+cell_offset, (new_df['x_end']*cell_length)+cell_offset, (new_df['y_end']*cell_length)+cell_offset, new_df['perc'], ax=ax,cmap='Reds', width=3, norm=norm)
    return fig, pcm

def get_similarities_movements(player_id, filename='seriea_2526', num_x_cells=5, num_y_cells=5):
    df = pd.read_pickle('grids_movements/' + filename + '_' + str(num_x_cells) + '_' + str(num_y_cells) + '.pkl')
    df = df.loc[df['played_matches'] >= played_matches_threshold]
    # df = pd.read_pickle('grids_movements_h/' + filename + '_' + str(num_x_cells) + '_' + str(num_y_cells) + '.pkl')
    player_names = df['player_name']
    player_ids = df['player_id']
    if 'league' in df.columns:
        player_leagues = df['league']
    # st.write(df)
    if 'teamName' in df.columns:
        player_teams = df['teamName']
    player_df = df.loc[df['player_id'] == player_id]
    if len(player_df)>1:
        st.error("Player has more than one team")
    else:
        player_df = player_df.iloc[0]
    grid = player_df['grid']
    grid_flat = grid.flatten()
    similarities = []
    grids = []
    for i in df.index:
        grid_compare = df.loc[i]['grid']
        grids.append(grid_compare)
        sim = cosine_similarity([grid_flat], [grid_compare])[0][0]
        similarities.append(sim)
    result_df = pd.DataFrame()
    result_df['player_name'] = player_names
    result_df['player_id'] = player_ids
    result_df['similarity'] = similarities
    result_df['grid'] = grids
    if 'league' in df:
        result_df['league'] = player_leagues
    if 'teamName' in df:
        result_df['team'] = player_teams
    result_df = result_df.sort_values(by=['similarity'], ascending=False)
    return result_df

def show_heatmaps(player_name, filename='top8_2526'):
    df = pd.read_pickle('grids/' + filename +'_30_30.pkl')
    df = df.loc[df['played_matches'] >= played_matches_threshold]
    df = df.sort_values(by=['player_name'])
    df["player_name"] = df["player_name"].apply(unidecode)

    # player_name = st.selectbox(
    #     "Search for a Player",
    #     options=df['player_name'],
    #     index=None,
    #     placeholder="Type a Player Name"
    # )

    # if player_name:
    row = df.loc[df['player_name'].str.lower() == player_name.lower()].iloc[0]
    player_id = row['player_id']
    fig, similarities = plot_players(player_id=player_id)
    st.pyplot(fig)

    top10 = similarities.iloc[1:11].reset_index(drop=True)
    st.subheader(f"Similarity Top 10 - {player_name}")

    col_left, col_right = st.columns(2)

    with col_left:
        for i, row in top10.iloc[0:5].iterrows():
            with st.container(border=True):
                # st.markdown(f"**#{i+1} — {row['player_name']} - {row['team']} - {row['position']}**")
                st.markdown(f"**#{i+1} — {row['player_name']} - {row['team']}**")
                st.progress(row['similarity'])
                st.caption(f"**{row['similarity']:.1%}**")
                with st.expander("Compare players"):
                    compare_players(player_name, row['player_name'], filename=filename, expander=True, type='heatmap')

    with col_right:
        for i, row in top10.iloc[5:10].iterrows():
            with st.container(border=True):
                # st.markdown(f"**#{i+1} — {row['player_name']} - {row['team']} - {row['position']}**")
                st.markdown(f"**#{i+1} — {row['player_name']} - {row['team']}**")
                st.progress(row['similarity'])
                st.caption(f"**{row['similarity']:.1%}**")
                with st.expander("Compare players"):
                    compare_players(player_name, row['player_name'], filename=filename, expander=True, type='heatmap')
                

def show_movements(num_x_cells=5, num_y_cells=5, top=20, filename='engitager_2526', player_name=None):
    # st.write(filename)
    df = pd.read_pickle('grids_movements/' + filename + '_' + str(num_x_cells) + '_' + str(num_y_cells) + '.pkl')
    df = df.loc[df['played_matches'] >= played_matches_threshold]
    # df = pd.read_pickle('grids_movements_h/' + filename + '_' + str(num_x_cells) + '_' + str(num_y_cells) + '.pkl')
    df = df.sort_values(by=['player_name'])
    df["player_name"] = df["player_name"].apply(unidecode)

    # st.write(df)
    # player_name = st.selectbox(
    #     "Search for a Player",
    #     options=df['player_name'],
    #     index=None,
    #     placeholder="Type a Player Name"
    # )

    # if player_name:
    player_df = df.loc[df['player_name'].str.lower() == player_name.lower()]
    if len(player_df) == 0:
        alt_name = find_alt_name(player_name=player_name)
        player_df = df.loc[df['player_name'].str.lower() == alt_name.lower()]
        if len(player_df) == 0:
            st.error(f"Player {player_name} not found!")
    if len(player_df)>1:
            st.error("Player has more than one team")
    else:
        row = player_df.iloc[0]
    # st.write(row)
    player_id = row['player_id']
    plot_players_movements(player_id=player_id, filename=filename, num_x_cells=num_x_cells, num_y_cells=num_y_cells, top=top)
        

def compare_heatmaps(player_name1, player_name2, df, num_x_cells, num_y_cells, expander=False):
    player_name1 = unidecode(player_name1)
    player_name2 = unidecode(player_name2)
    df['player_name'] = df['player_name'].apply(unidecode)
    compare_df1 = df.loc[df['player_name'].str.lower() == player_name1.lower()].iloc[0]
    grid_flat1 = np.array(compare_df1['grid'])
    grid1 = grid_flat1.reshape(num_x_cells,num_y_cells)

    compare_df2 = df.loc[df['player_name'].str.lower() == player_name2.lower()].iloc[0]
    grid_flat2 = np.array(compare_df2['grid'])
    grid2 = grid_flat2.reshape(num_x_cells,num_y_cells)

    matrices = [grid1, grid2]
    title1 = player_name1
    title2 = player_name2
    titles = [title1, title2]

    # pitch = VerticalPitch(pitch_type='opta', line_color='black', pitch_color='white', linewidth=1.5)
    pitch = VerticalPitch(pitch_type='opta', line_color='white', pitch_color='none', linewidth=1.5)

    if expander:
        fig, axes = plt.subplots(1, 2, figsize=(7, 7))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 12))

    fig.patch.set_alpha(0)
    for ax in axes:
        ax.patch.set_alpha(0)

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


        dark_blue_red_cmap = LinearSegmentedColormap.from_list(
            'dark_blue_red', ['#0e1117', '#e63946']
        )
        teal_cmap = LinearSegmentedColormap.from_list(
            'teal_cmap', ['#FFFFFF', '#0000FF']  # da teal chiarissimo a teal scuro
        )
        pcm = pitch.heatmap(stats, ax=ax, cmap=dark_blue_red_cmap, edgecolors='none',
                            alpha=0.75, zorder=1, vmin=vmin, vmax=vmax)
        pitch.draw(ax=ax)  # ridisegna le linee sopra
        ax.set_title(title, fontsize=12, color='white')
        

    # fig.colorbar(pcm, ax=axes, shrink=0.6, orientation='horizontal', pad=0.05)
    cbar = fig.colorbar(pcm, ax=axes, shrink=0.6, orientation='horizontal', pad=0.05)
    cbar.ax.xaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')
    st.pyplot(fig)

def compare_movements(player_name1, player_name2, df, num_x_cells, num_y_cells, top=20, expander=False):
    player_name1 = unidecode(player_name1)
    player_name2 = unidecode(player_name2)
    # st.write(player_name1)
    compare_df1 = df.loc[df['player_name'].str.lower() == player_name1.lower()]
    # st.write(compare_df1)
    row = compare_df1.iloc[0]
    grid_flat1 = np.array(row['grid'])
    grid1 = grid_flat1.reshape(num_x_cells*num_y_cells,num_y_cells*num_x_cells)

    compare_df2 = df.loc[df['player_name'].str.lower() == player_name2.lower()]
    row = compare_df2.iloc[0]
    grid_flat2 = np.array(row['grid'])
    grid2 = grid_flat2.reshape(num_x_cells*num_y_cells,num_y_cells*num_x_cells)

    all_top_vals = np.concatenate([
        top_percs(grid1, num_x_cells, num_y_cells, top),
        top_percs(grid2, num_x_cells, num_y_cells, top)
    ])
    vmin, vmax = all_top_vals.min(), all_top_vals.max()

    pitch = VerticalPitch(pitch_type='opta', line_color='white', pitch_color='none', linewidth=1.5)
    if expander:
        fig, axes = plt.subplots(1, 2, figsize=(7, 7))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 12))
    fig.patch.set_alpha(0)
    for ax in axes:
        pitch.draw(ax=ax)
        ax.patch.set_alpha(0)
    axes[0].set_title(player_name1, color='white')
    axes[1].set_title(player_name2, color='white')
    fig, pcm1 = plot_movements(grid1, fig, axes[0], pitch, num_x_cells, num_y_cells, vmin=vmin, vmax=vmax, top=top)
    fig, pcm2 = plot_movements(grid2, fig, axes[1], pitch, num_x_cells, num_y_cells, vmin=vmin, vmax=vmax, top=top)

    cbar = fig.colorbar(pcm1, ax=axes, shrink=0.6, orientation='horizontal', pad=0.05)
    cbar.ax.xaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')
    cbar.outline.set_edgecolor('white')
    
    st.pyplot(fig)

def compare_players(player_name1, player_name2, filename, num_x_cells_hea=30, num_y_cells_hea=30, num_x_cells_tou=5, num_y_cells_tou=5, top=20, expander=False, type='full'):
    # st.write(player_name1, player_name2)
    player_name1_hea = player_name1
    player_name1_tou = player_name1
    
    if type != 'movement':
        df_hea = pd.read_pickle('grids/' + filename + '_' + str(num_x_cells_hea) + '_' + str(num_y_cells_hea) + '.pkl')
        df_hea = df_hea.loc[df_hea['played_matches'] >= played_matches_threshold]
        df_hea["player_name"] = df_hea["player_name"].apply(unidecode)
        player_df = df_hea.loc[df_hea['player_name'].str.lower() == player_name.lower()]
        if len(player_df) == 0:
            alt_name = find_alt_name(player_name=player_name)
            player_df = df_hea.loc[df_hea['player_name'].str.lower() == alt_name.lower()]
            if len(player_df) == 0:
                st.error(f"Player {player_name} not found!")
            player_name1_hea = alt_name
        if len(player_df)>1:
            st.error("Player has more than one team")
    if type != 'heatmap':
        df_tou = pd.read_pickle('grids_movements/' + filename + '_' + str(num_x_cells_tou) + '_' + str(num_y_cells_tou) + '.pkl')
        df_tou = df_tou.loc[df_tou['played_matches'] >= played_matches_threshold]
        df_tou["player_name"] = df_tou["player_name"].apply(unidecode)
        player_df = df_tou.loc[df_tou['player_name'].str.lower() == player_name.lower()]
        if len(player_df) == 0:
            alt_name = find_alt_name(player_name=player_name)
            player_df = df_tou.loc[df_tou['player_name'].str.lower() == alt_name.lower()]
            if len(player_df) == 0:
                st.error(f"Player {player_name} not found!")
            player_name1_tou = alt_name
        if len(player_df)>1:
            st.error("Player has more than one team")
        
    
    

    if not expander:
        
        player_df = df_hea.loc[df_hea['player_name'] == player_name1_hea]
        if len(player_df)>1:
            st.error("Player has more than one team")
        else:
            player_df = player_df.iloc[0]
        player_id = player_df['player_id']
        sim_mov = get_similarities(player_id=player_id, filename=filename, num_x_cells=num_x_cells, num_y_cells=num_y_cells)
        sim_mov['player_name'] = sim_mov['player_name'].apply(unidecode)

        player_df = df_tou.loc[df_tou['player_name'] == player_name1_tou]
        if len(player_df)>1:
            st.error("Player has more than one team")
        else:
            player_df = player_df.iloc[0]
        player_id = player_df['player_id']
        sim_tou = get_similarities_movements(player_id=player_id, filename=filename, num_x_cells=num_x_cells, num_y_cells=num_y_cells)
        sim_tou['player_name'] = sim_tou['player_name'].apply(unidecode)

        # st.write(sim_mov)
        # st.write(sim_tou)

        merged = sim_mov.merge(
            sim_tou,
            on='player_name',
            suffixes=('_touch', '_movement')
        )
        merged['similarity_mixed'] = (merged['similarity_touch'] + merged['similarity_movement'])/2
        merged = merged.sort_values(by=['similarity_mixed'], ascending=False).reset_index()
        merged = merged.drop(columns=['index'])
        # st.write(merged)
        merged = merged.loc[merged['player_name'] == player_name2].iloc[0]
        hea_sim = merged['similarity_movement']
        tou_sim = merged['similarity_touch']
        mix_sim = merged['similarity_mixed']
        # st.write(merged)

        left, center, right = st.columns(3)

        with left:
            # st.write(f"Heatmap Similarity: {hea_sim:.1%}")
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="font-size: 3rem; font-weight: 700; color: #1f77b4; line-height: 1;">
                        {round(hea_sim*100, 2)}%
                    </div>
                    <div style="font-size: 0.9rem; color: gray; margin-top: 4px;">
                        <b>Heatmap</b> Similarity
                    </div>
                """,
                unsafe_allow_html=True
            )
        with center:
            # st.write(f"Movement Similarity: {tou_sim:.1%}")
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="font-size: 3rem; font-weight: 700; color: #1f77b4; line-height: 1;">
                        {round(tou_sim*100, 2)}%
                    </div>
                    <div style="font-size: 0.9rem; color: gray; margin-top: 4px;">
                        <b>Movement</b> Similarity
                    </div>
                """,
                unsafe_allow_html=True
            )
        with right:
            # st.write(f"Heatmap Similarity: {mix_sim:.1%}")
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="font-size: 3rem; font-weight: 700; color: #ff2b2b; line-height: 1;">
                        {round(mix_sim*100, 2)}%
                    </div>
                    <div style="font-size: 0.9rem; color: gray; margin-top: 4px;">
                        <b>Total</b> Similarity
                    </div>
                """,
                unsafe_allow_html=True
            )
        

    if type=='full':
        if not expander:
            st.subheader("Heatmap Comparison")
        compare_heatmaps(player_name1_hea, player_name2, df_hea, num_x_cells_hea, num_y_cells_hea, expander=expander)
        if not expander:
            st.subheader("Movements Comparison")
        compare_movements(player_name1_tou, player_name2, df_tou, num_x_cells_tou, num_y_cells_tou, top=top, expander=expander)
    elif type=='heatmap':
        compare_heatmaps(player_name1_hea, player_name2, df_hea, num_x_cells_hea, num_y_cells_hea, expander=expander)
    elif type=='movement':
        compare_movements(player_name1_tou, player_name2, df_tou, num_x_cells_tou, num_y_cells_tou, top=top, expander=expander)
        
def show_combined(player_name, filename, num_x_cells=5, num_y_cells=5, top=20):
    # st.write(player_name)
    df_hea = pd.read_pickle('grids/' + filename + '_30_30.pkl')
    df_hea = df_hea.loc[df_hea['played_matches'] >= played_matches_threshold]
    df_hea["player_name"] = df_hea["player_name"].apply(unidecode)
    df_tou = pd.read_pickle('grids_movements/' + filename + '_' + str(num_x_cells) + '_' + str(num_y_cells) + '.pkl')
    df_hea = df_hea.loc[df_hea['played_matches'] >= played_matches_threshold]
    df_tou["player_name"] = df_tou["player_name"].apply(unidecode)
    # st.write(df_hea)
    # st.write(df_tou)

    player_name_hea = player_name
    player_name_tou = player_name

    player_df = df_hea.loc[df_hea['player_name'].str.lower() == player_name.lower()]
    # st.write(player_name)
    # st.write(df_hea)
    if len(player_df) == 0:
        alt_name = find_alt_name(player_name=player_name)
        player_df = df_hea.loc[df_hea['player_name'].str.lower() == alt_name.lower()]
        if len(player_df) == 0:
            st.error(f"Player {player_name} not found!")
    if len(player_df)>1:
        st.error("Player has more than one team")
    else:
        row = player_df.iloc[0]
    player_id = row['player_id']
    sim_mov = get_similarities(player_id=player_id, filename=filename, num_x_cells=num_x_cells, num_y_cells=num_y_cells)

    player_df = df_tou.loc[df_tou['player_name'].str.lower() == player_name.lower()]
    if len(player_df) == 0:
        alt_name = find_alt_name(player_name=player_name)
        player_df = df_tou.loc[df_tou['player_name'].str.lower() == alt_name.lower()]
        if len(player_df) == 0:
            st.error(f"Player {player_name} not found!")
    if len(player_df)>1:
        st.error("Player has more than one team")
    else:
        row = player_df.iloc[0]
    player_id = row['player_id']
    sim_tou = get_similarities_movements(player_id=player_id, filename=filename, num_x_cells=num_x_cells, num_y_cells=num_y_cells)

    # st.write(player_name_hea, player_name_tou)
    # print(sim_mov.head())
    # print(sim_tou.head())

    merged = sim_mov.merge(
        sim_tou,
        on=['player_name', 'team', 'league'],
        suffixes=('_touch', '_movement')
    )
    merged['similarity_mixed'] = (merged['similarity_touch'] + merged['similarity_movement'])/2
    merged = merged.sort_values(by=['similarity_mixed'], ascending=False).reset_index()
    merged = merged.drop(columns=['index'])
    merged = merged[['player_name', 'league', 'team', 'position', 'similarity_mixed', 'similarity_movement', 'similarity_touch', 'player_id_movement', 'player_id_touch', 'grid']]
    # st.write(merged)

    top10 = merged.iloc[1:11].reset_index(drop=True)
    st.subheader(f"Similarity Top 10 - {player_name}")

    col_left, col_right = st.columns(2)

    with col_left:
        for i, row in top10.iloc[0:5].iterrows():
            with st.container(border=True):
                # st.markdown(f"**#{i+1} — {row['player_name']}**")
                st.markdown(f"**#{i+1} — {row['player_name']} - {row['team']}**")
                # st.markdown(f"**#{i+1} — {row['player_name']} - {row['team']} - {row['position']}**")
                st.progress(row['similarity_mixed'])
                st.caption(f"**{row['similarity_mixed']:.1%}** ({row['similarity_movement']:.1%} Heatmaps, {row['similarity_touch']:.1%} Movements)")
                with st.expander("Compare players"):
                    compare_players(player_name_hea, row['player_name'], filename=filename, num_x_cells_tou=num_x_cells, num_y_cells_tou=num_y_cells, top=top, expander=True)

    with col_right:
        for i, row in top10.iloc[5:10].iterrows():
            with st.container(border=True):
                # st.markdown(f"**#{i+1} — {row['player_name']}**")
                st.markdown(f"**#{i+1} — {row['player_name']} - {row['team']}**")
                # st.markdown(f"**#{i+1} — {row['player_name']} - {row['team']} - {row['position']}**")
                st.progress(row['similarity_mixed'])
                st.caption(f"**{row['similarity_mixed']:.1%}** ({row['similarity_movement']:.1%} Heatmaps, {row['similarity_touch']:.1%} Movements)")
                with st.expander("Compare players"):
                    compare_players(player_name_hea, row['player_name'], filename=filename, num_x_cells_tou=num_x_cells, num_y_cells_tou=num_y_cells, top=top, expander=True)


st.title("Player Similarity")
st.subheader("Search for a player in order to find the most similar players!")
st.write("Last Update: August 18th, 2026")

st.info("This project runs a similarity algorithm, based on player heatmaps. Note therefore that the similarity is based only on movement.  \nData are taken from the 2025/26 season of the top 5 European Leagues (England, Spain, Italy, Germany, France).")

df = pd.read_pickle('grids/top8_2526_30_30.pkl')
df = df.loc[df['played_matches'] >= played_matches_threshold]
df = df.sort_values(by=['player_name'])
df["player_name"] = df["player_name"].apply(unidecode)



# df['description'] = df['player_name'] + " - " + df['team']

# st.write(df)
player_name = st.selectbox(
    "Search for a Player",
    options=df['player_name'],
    index=None,
    placeholder="Type a Player Name"
)

if player_name:

    # player_string = str.split(player_name, " - ")
    # player_name = player_string[0]
    # player_team = player_string[1]
    # st.write(player_name)
    sim_choice = option_menu(None, ['Heatmap Similarity', 'Movement Similarity', 'Combined Similarity', 'Compare Players'],icons=['1-circle', '2-circle', '3-circle', '4-circle'], 
        default_index=0, orientation="horizontal")

    if sim_choice == 'Heatmap Similarity':
        show_heatmaps(player_name=player_name)
    elif sim_choice == 'Movement Similarity':
        wide_movements = st.checkbox('Wider Movements')
        more_movements = st.checkbox('Show More Movements')
        if wide_movements:
            num_x_cells = 5
            num_y_cells = 5
        else:
            num_x_cells = 7
            num_y_cells = 7
        if more_movements:
            top = 20
        else:
            top = 10
        show_movements(num_x_cells=num_x_cells, num_y_cells=num_y_cells, top=top, player_name=player_name, filename='top5_2526')
    elif sim_choice == 'Combined Similarity':
        wide_movements = st.checkbox('Wider Movements')
        more_movements = st.checkbox('Show More Movements')
        if not wide_movements:
            num_x_cells = 7
            num_y_cells = 7
        else:
            num_x_cells = 5
            num_y_cells = 5
        if not more_movements:
            top = 10
        else:
            top = 20
        show_combined(player_name, filename='top5_2526', num_x_cells=num_x_cells, num_y_cells=num_y_cells, top=top)
    elif sim_choice == 'Compare Players':
        filename = 'top5_2526'
        df = pd.read_pickle('grids/' + filename + '_30_30.pkl')
        df = df.loc[df['played_matches'] >= played_matches_threshold]
        df = df.sort_values(by=['player_name'])
        df["player_name"] = df["player_name"].apply(unidecode)
        player_name_compare = st.selectbox(
            "Search for the Comparison Player",
            options=df['player_name'],
            index=None,
            placeholder="Type a Player Name"
        )
        if player_name_compare:
            wide_movements = st.checkbox('Wider Movements')
            more_movements = st.checkbox('Show More Movements')
            if not wide_movements:
                num_x_cells = 7
                num_y_cells = 7
            else:
                num_x_cells = 5
                num_y_cells = 5
            if not more_movements:
                top = 10
            else:
                top = 20
            # compare_players(player_name, player_name_compare)
            compare_players(player_name, player_name_compare, filename=filename, num_x_cells_tou=num_x_cells, num_y_cells_tou=num_y_cells, top=top, expander=False)