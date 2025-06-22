import pandas as pd
import seaborn as sns
import streamlit as st 
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datasetPreprocessing import new_matchesDF, new_deliveriesDF


def app():
    st.markdown(
        '''
            <h1 style='text-align:center; color: #e8630a;'><strong>🌟EXPLORATORY DATA ANALYSIS🌟</strong></h1>
            <hr style="border-top: 3px solid #e8630a;">
        ''',
        unsafe_allow_html=True
    )

    #################################################################
    ################## MATCHES DATASET LOADING ######################
    #################################################################
    with st.expander('👉 Matches Dataset: 2008 - 2024'):
        st.write(new_matchesDF.head(5))

        if st.checkbox(label="View Code", key=0):
            st.code(
                '''
                    new_matchesDF = pd.read_csv('matches_2008-2024.csv')

                    new_matchesDF.columns = new_matchesDF.columns.str.strip()

                    st.write(new_matchesDF.head(5))
                    ''',
                language='python'
            )

    #################################################################
    ################## DELEVERY DATASET LOADING #####################
    #################################################################
    with st.expander('👉 Deliveries Dataset: 2008 - 2024'):
        st.write(new_deliveriesDF.head(5))

        if st.checkbox(label="View Code", key=1):
            st.code(
                '''
                    new_deliveriesDF = pd.read_csv(
                        'deliveries_2008-2024.csv')

                    new_deliveriesDF.columns = new_deliveriesDF.columns.str.strip()

                    st.write(new_deliveriesDF.head(5))
                    ''',
                language='python'
            )

    #################################################################
    ################## MATCHES PER SEASON ###########################
    #################################################################
    with st.expander("👉 Matches Per Season"):
        matches_per_season = new_matchesDF['season'].value_counts(
        ).sort_index()

        colors = px.colors.qualitative.Safe
        color_list = [
            colors[i % len(colors)] for i in range(len(matches_per_season))
        ]

        trace = go.Bar(
            x=matches_per_season.index,
            y=matches_per_season.values,
            text=matches_per_season.values,
            marker={'color': color_list}
        )

        layout = go.Layout(
            title="Matches Per Season",
            xaxis={'title': 'Season'},
            yaxis={'title': 'Number of Matches Played'}
        )

        fig = go.Figure(
            data=[trace],
            layout=layout
        )

        st.plotly_chart(
            fig,
            transparent=True,
            use_container_width=True
        )

        if st.checkbox(label="View Code", key=2):
            st.code(
                '''
                    matches_per_season = new_matchesDF['season'].value_counts(
                    ).sort_index()

                    colors = px.colors.qualitative.Safe

                    color_list = [colors[i % len(colors)]
                                    for i in range(len(matches_per_season))]

                    trace = go.Bar(x = matches_per_season.index,
                                    y = matches_per_season.values,
                                    marker={'color': color_list})

                    layout = go.Layout(title = "Matches Per Season",
                                        xaxis = {'title': 'Season'},
                                        yaxis = {'title': 'Number of Matches Played'})

                    fig = go.Figure(data = [trace], layout = layout)

                    st.plotly_chart(fig,
                                    transparent = True,
                                    use_container_width = True)
                    ''',
                language='python'
            )

    ####################################################################
    ########## Most Man of The Match Award Received By Players #########
    ###################################################################
    with st.expander("👉 Most POTM Awards"):
        top_20_POTM = new_matchesDF['player_of_match'].value_counts(
        ).iloc[:20].sort_values()

        fig = px.bar(
            x=top_20_POTM.values,
            y=top_20_POTM.index,
            labels={
                'y': 'Player',
                'x': 'POTM Awards'
            },
            color=top_20_POTM.index,
            text=top_20_POTM.values,
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig.update_layout(
            height=700,
            width=900,
        )

        st.plotly_chart(
            fig,
            transparent=True,
            use_container_width=True
        )

        if st.checkbox(label="View Code", key=3):
            st.code(
                '''
                    top_20_POTM = new_matchesDF['player_of_match'].value_counts(
                    ).iloc[:20].sort_values()
                    fig = px.bar(
                        y=top_20_POTM.index,
                        x=top_20_POTM.values,
                        labels={
                            'y': 'Player',
                            'x': 'POTM Awards'
                        },
                        color=top_20_POTM.index,
                        color_discrete_sequence=px.colors.qualitative.Safe
                    )
                    st.plotly_chart(fig,
                                    transparent = True,
                                    use_container_width = True)
                    ''',
                language='python'
            )

    ##########################################################################
    ################ Venues With Most Matches ################################
    ##########################################################################
    with st.expander("👉 Top 20 Venues With Most Matches"):
        top_20_venue = new_matchesDF['venue'].value_counts().iloc[:20]

        fig = px.bar(
            x=top_20_venue.index,
            y=top_20_venue.values,
            labels={
                'x': 'Venue',
                'y': 'Total Matches Played'
            },
            color=top_20_venue.index,
            text=top_20_venue.values,
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig.update_layout(
            height=700,
            width=900,
        )

        st.plotly_chart(
            fig,
            transparent=True,
            use_container_width=True
        )

        if st.checkbox(label="View Code", key=4):
            st.code(
                '''
                    top_20_venue = new_matchesDF['venue'].value_counts(
                    ).iloc[:20]
                    fig = px.bar(x=top_20_venue.index,
                                 y=top_20_venue.values,
                                 labels={
                                     'x':'Venue',
                                     'y':'Total Matches Played'
                                 },
                                 color=top_20_venue.index,
                                 color_discrete_sequence=px.colors.qualitative.Safe)
                    fig.update_layout(
                        height=700,
                        width=900,
                    )
                    st.plotly_chart(fig,
                                    transparent=True,
                                    use_container_width=True)
                    ''',
                language='python'
            )

    ###########################################################################
    ###################### Team With Most Match Wins ##########################
    ###########################################################################
    with st.expander('👉 Team With Most Match Wins'):
        teams_total_toss_win = new_matchesDF['winner'].value_counts(
        ).iloc[:].sort_values()

        fig = px.bar(
            y=teams_total_toss_win.index,
            x=teams_total_toss_win.values,
            labels={
                'y': 'Total Matches Won',
                'x': 'Team'
            },
            color=teams_total_toss_win.index,
            text=teams_total_toss_win.values,
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig.update_layout(
            height=700,
            width=900,
        )

        st.plotly_chart(
            fig,
            transparent=True,
            use_container_width=True
        )

        if st.checkbox(label="View Code", key=5):
            st.code(
                '''
                    teams_total_toss_win = new_matchesDF['winner'].value_counts(
                    ).iloc[:20].sort_values()
                    fig = px.bar(y=teams_total_toss_win.index,
                                 x=teams_total_toss_win.values,
                                 labels={
                                     'y':'Team',
                                     'x':'Total Wins'
                                 },
                                 color=teams_total_toss_win.index,
                                 color_discrete_sequence=px.colors.qualitative.Safe
                    fig.update_layout(
                        height=700,
                        width=900,
                    )
                    st.plotly_chart(fig,
                                    transparent=True,
                                    use_container_width=True)
            ''',
                language='python'
            )

    ##################################################################
    #################### Team With Most Toss Wins ####################
    ##################################################################
    with st.expander('👉 Team With Most Toss Wins'):
        teams_toss_win_count = new_matchesDF['toss_winner'].value_counts()
        
        fig = px.bar(
            x=teams_toss_win_count.index,
            y=teams_toss_win_count.values,
            title='Toss Winners',
            labels={
                'y': 'Total Tosses Won',
                'x': 'Team'
            },
            color=teams_toss_win_count.index,
            text=teams_toss_win_count.values,
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        st.plotly_chart(
            fig,
            transparent=True,
            use_container_width=True
        )

        if st.checkbox(label="View Code", key=6):
            st.code(
                '''
                    teams_toss_win_count = matches_team['toss_winner'].value_counts(
                    )

                    fig = px.bar(x=teams_toss_win_count.index,
                                 y=teams_toss_win_count.values,
                                 title='Toss Winners'
                                 color=teams_toss_win_count.index,
                                 color_discrete_sequence=px.colors.qualitative.Safe)

                    st.plotly_chart(fig,
                                    transparent=True,
                                    use_container_width=True)
                ''',
                language='python'
            )

    #####################################################################
    ########### Win Percentage of Team after Winning The Toss  ##########
    #####################################################################
    with st.expander('👉 Win Percentage of Team after Winning The Toss'):
        fig = plt.figure(figsize=(15, 8))

        win_percentage_after_toss = round(
            (new_matchesDF[
                new_matchesDF['toss_winner'] == new_matchesDF['winner']
            ]['winner'].value_counts() / new_matchesDF['toss_winner'].value_counts()
            ) * 100
        ).sort_values(ascending=False)

        explode = [0.1] + [0] * (len(win_percentage_after_toss) - 1)

        plt.rcParams.update(
            {
                'text.color': "white",
                'axes.labelcolor': "black"
            }
        )

        win_percentage_after_toss.plot(
            kind='pie',
            autopct='%1.1f%%',
            explode=explode,
            shadow=True,
            startangle=90
        )

        plt.ylabel('')
        plt.title('Win Percentage of Team after Winning The Toss')

        col1, col2 = st.columns([3, 2])

        with col1:
            st.pyplot(
                fig,
                transparent=True
            )

        with col2:
            win_percentage_df = win_percentage_after_toss.reset_index()
            win_percentage_df.columns = ['Team', 'Win % After Toss']

            st.write('### Overall Record:')
            st.dataframe(win_percentage_after_toss, width=500, height=575)

        if st.checkbox(label="View Code", key=7):
            st.code(
                '''
                    win_percentage_after_toss = round(
                        (
                            new_matchesDF[new_matchesDF['toss_winner']
                                        == new_matchesDF['winner']]
                            ['winner'].value_counts() / \
                                                    new_matchesDF['toss_winner'].value_counts()
                        ) * 100
                    ).sort_values(ascending=False)

                    explode = [0.1] + [0] * (len(win_percentage_after_toss) - 1)

                    win_percentage_after_toss.plot(kind='pie',
                                                autopct='%1.1f%%',
                                                explode=explode,
                                                shadow=True,
                                                startangle=90)

                    plt.rcParams.update({'text.color': "white",
                                        'axes.labelcolor': "black"})

                    fig = plt.figure(figsize=(15, 8))

                    plt.ylabel('')

                    plt.title('Win Percentage of Team after Winning The Toss')

                    st.pyplot(fig, transparent=True)
                ''',
                language='python'
            )

    #####################################################################
    ############### Teams Winning Both Toss and Matches #################
    #####################################################################
    with st.expander('👉 Teams Winning Both Toss and Matches Since 2008'):
        toss_Match_winner = new_matchesDF[
            new_matchesDF['toss_winner'] == new_matchesDF['winner']
        ]['winner'].value_counts()

        colors = px.colors.qualitative.Safe
        color_list = colors[:len(toss_Match_winner)]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=toss_Match_winner.index,
                y=toss_Match_winner.values,
                text=toss_Match_winner.values,
                textposition='outside',
                marker_color=color_list,
            )
        )

        fig.update_layout(
            title="Teams Winning Both Toss and Matches",
            xaxis_title="Teams",
            yaxis_title="Total Matches Won",
            xaxis=dict(tickangle=-45),
            height=800,
            width=1200,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:
            st.write('### Records:')
            st.dataframe(toss_Match_winner, width=400, height=570)

        if st.checkbox(label="View Code", key=8):
            st.code(
                '''
                        toss_Match_winner = new_matchesDF[new_matchesDF['toss_winner']
                                          == new_matchesDF['winner']]['winner'].value_counts()

                        colors = px.colors.qualitative.Safe
                        color_list = colors[:len(toss_Match_winner)]

                        fig = go.Figure()

                        fig.add_trace(go.Bar(
                            x=toss_Match_winner.index,
                            y=toss_Match_winner.values,
                            text=toss_Match_winner.values,
                            textposition='outside',
                            marker_color=color_list,
                        ))

                        fig.update_layout(
                            title="Teams Winning Both Toss and Matches",
                            xaxis_title="Teams",
                            yaxis_title="Total Matches Won",
                            xaxis=dict(tickangle=-45),
                            height=800,
                            width=1200,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )

                        st.plotly_chart(fig,
                                use_container_width=True)
                ''',
                language='python'
            )

    ############################################################################
    ################## Top 20 Players With Most Runs ###########################
    ############################################################################
    with st.expander('👉 Top 20 Players With Most Runs'):
        top_20_run_scorer = new_deliveriesDF.groupby(
            'batter')['batsman_runs'].sum().sort_values(ascending=False)[:20]

        top_20_run_scorer_df = top_20_run_scorer.reset_index()
        top_20_run_scorer_df.columns = ['Player', 'Runs']

        fig = px.bar(
            top_20_run_scorer_df,
            x='Runs',
            y='Player',
            labels={
                'Player': 'Player',
                'Runs': 'Total Runs'
            },
            color='Player',
            text='Runs',
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig.update_layout(
            height=600,
            width=800,
            yaxis_title='Player',
            xaxis_title='Total Runs',
            yaxis=dict(categoryorder='total ascending')
        )

        fig.update_traces(textposition='outside')

        st.plotly_chart(
            fig,
            transparent=True,
            use_container_width=True
        )

        if st.checkbox(label="View Code", key=9):
            st.code(
                '''
                    top_20_run_scorer = new_deliveriesDF.groupby(
                                        'batter')['batsman_runs'].sum().sort_values(ascending=False)[:20]

                    top_20_run_scorer_df = top_20_run_scorer.reset_index()
                    top_20_run_scorer_df.columns = ['Player', 'Runs']

                    fig = px.bar(
                        top_20_run_scorer_df,
                        x='Runs',
                        y='Player',
                        labels={
                            'Player': 'Player',
                            'Runs': 'Total Runs'
                        },
                        color='Player',
                        text='Runs',
                        color_discrete_sequence=px.colors.qualitative.Safe
                    )

                    fig.update_layout(
                        height=600,
                        width=800,
                        yaxis_title='Player',
                        xaxis_title='Total Runs',
                        yaxis=dict(categoryorder='total ascending')
                    )

                    fig.update_traces(textposition='outside')

                    st.plotly_chart(fig, transparent=True,
                                    use_container_width=True)

                ''',
                language='python'
            )

    #############################################################################
    ###############           MOST EXPENSIVE BOWLERS              ###############
    #############################################################################

    def plot_bar(data, title, xlabel, ylabel, col_width=2, rows=1):
        fig = plt.figure(figsize=(10, 6))

        ax = sns.barplot(
            x='bowler',
            y='total_runs',
            data=data[:10],
            palette='viridis',
            hue=None,
            legend=False
        )

        plt.yticks(
            rotation=90,
            fontsize=10,
            color='white'
        )

        plt.title(title,color='white')
        plt.xticks(
            rotation=45,
            ha='right',
            color='white'
        )
        plt.xlabel(xlabel,color='white')
        plt.ylabel(ylabel,color='white')

        for p in ax.patches:
            ax.annotate(
                format(p.get_height(), '.1f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 10),
                textcoords='offset points'
            )

        st.pyplot(fig, transparent=True)

    with st.expander("👉 Most Expensive Bowler"):
        st.write("> Overall Most Expensive Bowler:")
        col1, col2 = st.columns([3, 2])

        with col1:
            overall = new_deliveriesDF.groupby('bowler')['total_runs'].agg('sum').reset_index().sort_values('total_runs',
                                                                                                            ascending=False).head(30)
            plot_bar(
                overall,
                'Overall Most Expensive Bowler',
                'Bowler',
                'Total Runs'
            )

        with col2:
            st.dataframe(
                overall,
                width=400,
                height=400
            )

        st.write('> Most Expensive Bowler in 1st Over:')
        col3, col4 = st.columns([3, 2])

        with col3:
            first_over = new_deliveriesDF[new_deliveriesDF['over'] == 0]

            group = first_over.groupby('bowler')['total_runs'].agg('sum').reset_index().sort_values('total_runs',
                                                                                                    ascending=False).head(30)
            plot_bar(
                group,
                'Most Expensive Bowler in 1st Over',
                'Bowler',
                'Total Runs'
            )

        with col4:
            st.dataframe(
                group,
                width=400,
                height=400
            )

        st.write('> Most Expensive Bowler in 20th Over:')
        col5, col6 = st.columns([3, 2])

        with col5:
            twenty_over = new_deliveriesDF[new_deliveriesDF['over'] == 19]

            group = twenty_over.groupby('bowler')['total_runs'].agg('sum').reset_index().sort_values('total_runs',
                                                                                                     ascending=False).head(30)
            plot_bar(
                group,
                'Most Expensive Bowler in 20th Over',
                'Bowler',
                'Total Runs'
            )

        with col6:
            st.dataframe(
                group,
                width=400,
                height=400
            )

        if st.checkbox(label="View Code", key=10):
            st.code(
                '''
                    overall = new_deliveriesDF.groupby('bowler')['total_runs'].agg('sum').reset_index().sort_values('total_runs',
                                                                                                                                ascending=False).head(30)

                    fig = plt.figure(figsize=(10, 6))

                    top_bowlers = overall.head(10)

                    ax = sns.barplot(x='bowler',
                                    y='total_runs',
                                    data=top_bowlers,
                                    palette='viridis')

                    plt.xticks(rotation=90,
                                fontsize=10)

                    ax.bar_label(ax.containers[0])

                    plt.title('Overall Most Expensive Bowler')
                    plt.xlabel('Bowler')
                    plt.ylabel('Total Runs')

                    st.pyplot(fig, transparent=True)
                ''',
                language='python'
            )
     #######################################################################
    #####       Most sixes by a player since 2008  #########################
    #######################################################################
    with st.expander('👉 Most Sixes By A Player Since 2008'):
        most_sixes = new_deliveriesDF[new_deliveriesDF['batsman_runs'] == 6]['batter'].value_counts().head(20)

        fig = px.bar(
            x=most_sixes.index,
            y=most_sixes.values,
            labels={
                'x': 'Player',
                'y': 'Total Sixes'
            },
            color=most_sixes.index,
            text=most_sixes.values,
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig.update_layout(
            height=600,
            width=800,
            yaxis_title='Player',
            xaxis_title='Total Sixes',
            yaxis=dict(categoryorder='total ascending')
        )

        fig.update_traces(textposition='outside')

        st.plotly_chart(
            fig,
            transparent=True,
            use_container_width=True
        )

        if st.checkbox(label="View Code", key=11):
            st.code(
                '''
                    most_sixes = new_deliveriesDF[new_deliveriesDF['batsman_runs'] == 6]['batter'].value_counts().head(20)

                    fig = px.bar(x=most_sixes.index,
                                 y=most_sixes.values,
                                 labels={
                                     'x':'Player',
                                     'y':'Total Sixes'
                                 },
                                 color=most_sixes.index,
                                 text=most_sixes.values,
                                 color_discrete_sequence=px.colors.qualitative.Safe)

                    fig.update_layout(
                        height=600,
                        width=800,
                        yaxis_title='Player',
                        xaxis_title='Total Sixes',
                        yaxis=dict(categoryorder='total ascending')
                    )

                    fig.update_traces(textposition='outside')

                    st.plotly_chart(fig, transparent=True, use_container_width=True)
                ''',
                language='python'
            )
     #######################################################################
    #####       most sixes by a player in powerplay(0-6) Since 2008 #######
    #######################################################################
    with st.expander('👉 Most Sixes By A Player In Powerplay (0-6) Since 2008'):
        powerplay_sixes = new_deliveriesDF[
            (new_deliveriesDF['batsman_runs'] == 6) &
            (new_deliveriesDF['over'] < 6)
        ]['batter'].value_counts().head(20)

        fig = px.bar(
            x=powerplay_sixes.index,
            y=powerplay_sixes.values,
            labels={
                'x': 'Player',
                'y': 'Total Sixes'
            },
            color=powerplay_sixes.index,
            text=powerplay_sixes.values,
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig.update_layout(
            height=600,
            width=800,
            yaxis_title='Player',
            xaxis_title='Total Sixes',
            yaxis=dict(categoryorder='total ascending')
        )

        fig.update_traces(textposition='outside')

        st.plotly_chart(
            fig,
            transparent=True,
            use_container_width=True
        )

        if st.checkbox(label="View Code", key=12):
            st.code(
                '''
                    powerplay_sixes = new_deliveriesDF[(new_deliveriesDF['batsman_runs'] == 6) & (new_deliveriesDF['over'] < 6)]['batter'].value_counts().head(20)

                    fig = px.bar(x=powerplay_sixes.index,
                                 y=powerplay_sixes.values,
                                 labels={
                                     'x':'Player',
                                     'y':'Total Sixes'
                                 },
                                 color=powerplay_sixes.index,
                                 text=powerplay_sixes.values,
                                 color_discrete_sequence=px.colors.qualitative.Safe)

                    fig.update_layout(
                        height=600,
                        width=800,
                        yaxis_title='Player',
                        xaxis_title='Total Sixes',
                        yaxis=dict(categoryorder='total ascending')
                    )

                    fig.update_traces(textposition='outside')

                    st.plotly_chart(fig, transparent=True, use_container_width=True)
                ''',
                language='python'
            )
    #######################################################################
    #####       most sixes by a player in death(16-20) Since 2008 #######
    #######################################################################
    with st.expander('👉 Most Sixes By A Player In Death Overs (16-20) Since 2008'):
        death_sixes = new_deliveriesDF[
            (new_deliveriesDF['batsman_runs'] == 6) &
            (new_deliveriesDF['over'] >= 16)
        ]['batter'].value_counts().head(20)

        fig = px.bar(
            x=death_sixes.index,
            y=death_sixes.values,
            labels={
                'x': 'Player',
                'y': 'Total Sixes'
            },
            color=death_sixes.index,
            text=death_sixes.values,
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig.update_layout(
            height=600,
            width=800,
            yaxis_title='Player',
            xaxis_title='Total Sixes',
            yaxis=dict(categoryorder='total ascending')
        )

        fig.update_traces(textposition='outside')

        st.plotly_chart(
            fig,
            transparent=True,
            use_container_width=True
        )

        if st.checkbox(label="View Code", key=13):
            st.code(
                '''
                    death_sixes = new_deliveriesDF[(new_deliveriesDF['batsman_runs'] == 6) & (new_deliveriesDF['over'] >= 16)]['batter'].value_counts().head(20)

                    fig = px.bar(x=death_sixes.index,
                                 y=death_sixes.values,
                                 labels={
                                     'x':'Player',
                                     'y':'Total Sixes'
                                 },
                                 color=death_sixes.index,
                                 text=death_sixes.values,
                                 color_discrete_sequence=px.colors.qualitative.Safe)

                    fig.update_layout(
                        height=600,
                        width=800,
                        yaxis_title='Player',
                        xaxis_title='Total Sixes',
                        yaxis=dict(categoryorder='total ascending')
                    )

                    fig.update_traces(textposition='outside')

                    st.plotly_chart(fig, transparent=True, use_container_width=True)
                ''',
                language='python'
            )
    #######################################################################
    #####  most sixes by a player in middle overs(7-15) Since 2008 #######
    #######################################################################
    with st.expander('👉 Most Sixes By A Player In Middle Overs (7-15) Since 2008'):
        middle_sixes = new_deliveriesDF[
            (new_deliveriesDF['batsman_runs'] == 6) &
            (new_deliveriesDF['over'] >= 6) &
            (new_deliveriesDF['over'] < 16)
        ]['batter'].value_counts().head(20)

        fig = px.bar(
            x=middle_sixes.index,
            y=middle_sixes.values,
            labels={
                'x': 'Player',
                'y': 'Total Sixes'
            },
            color=middle_sixes.index,
            text=middle_sixes.values,
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig.update_layout(
            height=600,
            width=800,
            yaxis_title='Player',
            xaxis_title='Total Sixes',
            yaxis=dict(categoryorder='total ascending')
        )

        fig.update_traces(textposition='outside')

        st.plotly_chart(
            fig,
            transparent=True,
            use_container_width=True
        )

        if st.checkbox(label="View Code", key=14):
            st.code(
                '''
                    middle_sixes = new_deliveriesDF[(new_deliveriesDF['batsman_runs'] == 6) & (new_deliveriesDF['over'] >= 6) & (new_deliveriesDF['over'] < 16)]['batter'].value_counts().head(20)

                    fig = px.bar(x=middle_sixes.index,
                                 y=middle_sixes.values,
                                 labels={
                                     'x':'Player',
                                     'y':'Total Sixes'
                                 },
                                 color=middle_sixes.index,
                                 text=middle_sixes.values,
                                 color_discrete_sequence=px.colors.qualitative.Safe)

                    fig.update_layout(
                        height=600,
                        width=800,
                        yaxis_title='Player',
                        xaxis_title='Total Sixes',
                        yaxis=dict(categoryorder='total ascending')
                    )

                    fig.update_traces(textposition='outside')

                    st.plotly_chart(fig, transparent=True, use_container_width=True)
                ''',
                language='python'
            )
    #######################################################################
    #####       Overwise Average Runs For Each Team Since 2008      #######
    #######################################################################
    with st.expander('👉 Overwise Average Runs For Each Team Since 2008'):
        corr = new_deliveriesDF.pivot_table(
            values='total_runs',
            index='batting_team',
            columns='over',
            aggfunc='mean'
        ).fillna(0) * 6

        for over in range(0, 20):
            if over not in corr.columns:
                corr[over] = 0

        corr = corr[sorted(corr.columns)]

        corr_transposed = corr.T

        fig = px.imshow(
            corr_transposed,
            color_continuous_scale="viridis",
            labels=dict(
                x="Team",
                y="Over",
                color="Runs"
            ),
            x=corr_transposed.columns,
            y=corr_transposed.index
        )

        fig.update_yaxes(
            tickvals=list(range(1, 21)),
            title_text='Overs',
            title_font_size=16
        )

        fig.update_xaxes(
            title_text='Teams',
            title_font_size=16,
            tickangle=45
        )

        st.plotly_chart(
            fig,
            transparent=True,
            use_container_width=True
        )
    #######################################################################
    #####       most fours by a player in powerplay(0-6) Since 2008 #######
    #######################################################################
    with st.expander('👉 Most Fours By A Player In Powerplay (0-6) Since 2008'):
        powerplay_fours = new_deliveriesDF[
            (new_deliveriesDF['batsman_runs'] == 4) &
            (new_deliveriesDF['over'] < 6)
        ]['batter'].value_counts().head(20)

        fig = px.bar(
            x=powerplay_fours.index,
            y=powerplay_fours.values,
            labels={
                'x': 'Player',
                'y': 'Total Fours'
            },
            color=powerplay_fours.index,
            text=powerplay_fours.values,
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig.update_layout(
            height=600,
            width=800,
            yaxis_title='Player',
            xaxis_title='Total Fours',
            yaxis=dict(categoryorder='total ascending')
        )

        fig.update_traces(textposition='outside')

        st.plotly_chart(
            fig,
            transparent=True,
            use_container_width=True
        )

        if st.checkbox(label="View Code", key=15):
            st.code(
                '''
                    powerplay_fours = new_deliveriesDF[(new_deliveriesDF['batsman_runs'] == 4) & (new_deliveriesDF['over'] < 6)]['batter'].value_counts().head(20)

                    fig = px.bar(x=powerplay_fours.index,
                                 y=powerplay_fours.values,
                                 labels={
                                     'x':'Player',
                                     'y':'Total Fours'
                                 },
                                 color=powerplay_fours.index,
                                 text=powerplay_fours.values,
                                 color_discrete_sequence=px.colors.qualitative.Safe)

                    fig.update_layout(
                        height=600,
                        width=800,
                        yaxis_title='Player',
                        xaxis_title='Total Fours',
                        yaxis=dict(categoryorder='total ascending')
                    )

                    fig.update_traces(textposition='outside')

                    st.plotly_chart(fig, transparent=True, use_container_width=True)
                ''',
                language='python'
            )
    #######################################################################
    #####       most fours by a player in middle overs(7-16) Since 2008 #######
    #######################################################################
    with st.expander('👉 Most Fours By A Player In Middle Overs (7-16) Since 2008'):
        middle_fours = new_deliveriesDF[
            (new_deliveriesDF['batsman_runs'] == 4) &
            (new_deliveriesDF['over'] >= 6) &
            (new_deliveriesDF['over'] < 16)
        ]['batter'].value_counts().head(20)

        fig = px.bar(
            x=middle_fours.index,
            y=middle_fours.values,
            labels={
                'x': 'Player',
                'y': 'Total Fours'
            },
            color=middle_fours.index,
            text=middle_fours.values,
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig.update_layout(
            height=600,
            width=800,
            yaxis_title='Player',
            xaxis_title='Total Fours',
            yaxis=dict(categoryorder='total ascending')
        )

        fig.update_traces(textposition='outside')

        st.plotly_chart(
            fig,
            transparent=True,
            use_container_width=True
        )

        if st.checkbox(label="View Code", key=16):
            st.code(
                '''
                    middle_fours = new_deliveriesDF[(new_deliveriesDF['batsman_runs'] == 4) & (new_deliveriesDF['over'] >= 6) & (new_deliveriesDF['over'] < 16)]['batter'].value_counts().head(20)

                    fig = px.bar(x=middle_fours.index,
                                 y=middle_fours.values,
                                 labels={
                                     'x':'Player',
                                     'y':'Total Fours'
                                 },
                                 color=middle_fours.index,
                                 text=middle_fours.values,
                                 color_discrete_sequence=px.colors.qualitative.Safe)

                    fig.update_layout(
                        height=600,
                        width=800,
                        yaxis_title='Player',
                        xaxis_title='Total Fours',
                        yaxis=dict(categoryorder='total ascending')
                    )

                    fig.update_traces(textposition='outside')

                    st.plotly_chart(fig, transparent=True, use_container_width=True)
                ''',
                language='python'
            )
    #######################################################################
    #####       most fours by a player in death(16-20) Since 2008 #######
    #######################################################################
    with st.expander('👉 Most Fours By A Player In Death Overs (16-20) Since 2008'):
        death_fours = new_deliveriesDF[
            (new_deliveriesDF['batsman_runs'] == 4) &
            (new_deliveriesDF['over'] >= 16)
        ]['batter'].value_counts().head(20)

        fig = px.bar(
            x=death_fours.index,
            y=death_fours.values,
            labels={
                'x': 'Player',
                'y': 'Total Fours'
            },
            color=death_fours.index,
            text=death_fours.values,
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig.update_layout(
            height=600,
            width=800,
            yaxis_title='Player',
            xaxis_title='Total Fours',
            yaxis=dict(categoryorder='total ascending')
        )

        fig.update_traces(textposition='outside')

        st.plotly_chart(
            fig,
            transparent=True,
            use_container_width=True
        )

        if st.checkbox(label="View Code", key=17):
            st.code(
                '''
                    death_fours = new_deliveriesDF[(new_deliveriesDF['batsman_runs'] == 4) & (new_deliveriesDF['over'] >= 16)]['batter'].value_counts().head(20)

                    fig = px.bar(x=death_fours.index,
                                 y=death_fours.values,
                                 labels={
                                     'x':'Player',
                                     'y':'Total Fours'
                                 },
                                 color=death_fours.index,
                                 text=death_fours.values,
                                 color_discrete_sequence=px.colors.qualitative.Safe)

                    fig.update_layout(
                        height=600,
                        width=800,
                        yaxis_title='Player',
                        xaxis_title='Total Fours',
                        yaxis=dict(categoryorder='total ascending')
                    )

                    fig.update_traces(textposition='outside')

                    st.plotly_chart(fig, transparent=True, use_container_width=True)
                ''',
                language='python'
            )
    #######################################################################
    #####  most sixes in a season by a batsman                #######
    #######################################################################
    with st.expander('👉 Most Sixes In A Season By A Batsman'):
        match_season_map = new_matchesDF.set_index("id")["season"].to_dict()

        # Add new 'season' column to deliveries
        new_deliveriesDF["season"] = new_deliveriesDF["match_id"].map(match_season_map)

        most_sixes_season = new_deliveriesDF[new_deliveriesDF['batsman_runs'] == 6].groupby(
            ['batter', 'season']
        )['batsman_runs'].count().reset_index()

        most_sixes_season = most_sixes_season.sort_values(
            by='batsman_runs',
            ascending=False
        ).head(20)

        fig = px.bar(
            most_sixes_season,
            x='batsman_runs',
            y='batter',
            color='season',
            labels={
                'batter': 'Player',
                'batsman_runs': 'Total Sixes'
            },
            text='batsman_runs',
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig.update_layout(
            height=600,
            width=800,
            yaxis_title='Player',
            xaxis_title='Total Sixes',
            yaxis=dict(categoryorder='total ascending')
        )

        fig.update_traces(textposition='outside')

        st.plotly_chart(
            fig,
            transparent=True,
            use_container_width=True
        )
    #######################################################################
    #####  batsman with most runs through boundaries      #######
    #######################################################################
    with st.expander('👉 Batsman With Most Runs Through Boundaries'):
        boundaries = new_deliveriesDF[
            (new_deliveriesDF['batsman_runs'] == 4) |
            (new_deliveriesDF['batsman_runs'] == 6)
        ]

        boundaries['boundary_runs'] = boundaries['batsman_runs'].apply(
            lambda x: x if x in [4, 6] else 0
        )

        boundary_runs = boundaries.groupby('batter')['boundary_runs'].sum().sort_values(
            ascending=False
        ).head(20)

        fig = px.bar(
            x=boundary_runs.index,
            y=boundary_runs.values,
            labels={
                'x': 'Player',
                'y': 'Total Boundary Runs'
            },
            color=boundary_runs.index,
            text=boundary_runs.values,
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig.update_layout(
            height=600,
            width=800,
            yaxis_title='Player',
            xaxis_title='Total Boundary Runs',
            yaxis=dict(categoryorder='total ascending')
        )

        fig.update_traces(textposition='outside')

        st.plotly_chart(
            fig,
            transparent=True,
            use_container_width=True
        )

        if st.checkbox(label="View Code", key=18):
            st.code(
                '''
                    boundaries = new_deliveriesDF[(new_deliveriesDF['batsman_runs'] == 4) | (new_deliveriesDF['batsman_runs'] == 6)]

                    boundaries['boundary_runs'] = boundaries['batsman_runs'].apply(lambda x: x if x in [4, 6] else 0)

                    boundary_runs = boundaries.groupby('batter')['boundary_runs'].sum().sort_values(ascending=False).head(20)

                    fig = px.bar(x=boundary_runs.index,
                                 y=boundary_runs.values,
                                 labels={
                                     'x':'Player',
                                     'y':'Total Boundary Runs'
                                 },
                                 color=boundary_runs.index,
                                 text=boundary_runs.values,
                                 color_discrete_sequence=px.colors.qualitative.Safe)

                    fig.update_layout(
                        height=600,
                        width=800,
                        yaxis_title='Player',
                        xaxis_title='Total Boundary Runs',
                        yaxis=dict(categoryorder='total ascending')
                    )

                    fig.update_traces(textposition='outside')

                    st.plotly_chart(fig, transparent=True, use_container_width=True
                ''',
                language='python'
            )
    #######################################################################
    ##### highest scores in ipl history by players                  #######
    #######################################################################    
    with st.expander('👉 Highest Scores In IPL History By Players'):
        highest_scores = new_deliveriesDF.groupby('batter')['total_runs'].sum().sort_values(
            ascending=False
        ).head(20)

        highest_scores_df = highest_scores.reset_index()
        highest_scores_df.columns = ['Player', 'Total Runs']

        fig = px.bar(
            highest_scores_df,
            x='Total Runs',
            y='Player',
            labels={
                'Player': 'Player',
                'Total Runs': 'Total Runs'
            },
            color='Player',
            text='Total Runs',
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig.update_layout(
            height=600,
            width=800,
            yaxis_title='Player',
            xaxis_title='Total Runs',
            yaxis=dict(categoryorder='total ascending')
        )

        fig.update_traces(textposition='outside')

        st.plotly_chart(
            fig,
            transparent=True,
            use_container_width=True
        )

        if st.checkbox(label="View Code", key=19):
            st.code(
                '''
                    highest_scores = new_deliveriesDF.groupby('batter')['total_runs'].sum().sort_values(ascending=False).head(20)

                    highest_scores_df = highest_scores.reset_index()
                    highest_scores_df.columns = ['Player', 'Total Runs']

                    fig = px.bar(highest_scores_df,
                                 x='Total Runs',
                                 y='Player',
                                 labels={
                                     'Player': 'Player',
                                     'Total Runs': 'Total Runs'
                                 },
                                 color='Player',
                                 text='Total Runs',
                                 color_discrete_sequence=px.colors.qualitative.Safe)

                    fig.update_layout(
                        height=600,
                        width=800,
                        yaxis_title='Player',
                        xaxis_title='Total Runs',
                        yaxis=dict(categoryorder='total ascending')
                    )

                    fig.update_traces(textposition='outside')

                    st.plotly_chart(fig, transparent=True, use_container_width=True)
                ''',
                language='python'
            )
    #######################################################################
    #####     highest runs by a player in a season             #######
    #######################################################################
    with st.expander('👉 Highest Runs By A Player In A Season'):
        match_season_map = new_matchesDF.set_index("id")["season"].to_dict()

# Add new 'season' column to deliveries
        new_deliveriesDF["season"] = new_deliveriesDF["match_id"].map(match_season_map)
        highest_runs_season = new_deliveriesDF.groupby(
            ['batter', 'season']
        )['batsman_runs'].sum().reset_index()

        highest_runs_season = highest_runs_season.sort_values(
            by='batsman_runs',
            ascending=False
        ).head(20)

        fig = px.bar(
            highest_runs_season,
            x='batsman_runs',
            y='batter',
            color='season',
            labels={
                'batter': 'Player',
                'batsman_runs': 'Total Runs'
            },
            text='batsman_runs',
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig.update_layout(
            height=600,
            width=800,
            yaxis_title='Player',
            xaxis_title='Total Runs',
            yaxis=dict(categoryorder='total ascending')
        )

        fig.update_traces(textposition='outside')

        st.plotly_chart(
            fig,
            transparent=True,
            use_container_width=True
        )
    
    #######################################################################
    #####  most overall strikerate of batsman  with more than 1000 runs #######
    #######################################################################
    with st.expander('👉 Batsman With Most Overall Strike Rate (More Than 1000 Runs)'):
        overall_strike_rate = new_deliveriesDF.groupby('batter').agg(
            total_runs=('batsman_runs', 'sum'),
            balls_faced=('ball', 'count')
        ).reset_index()

        overall_strike_rate['strike_rate'] = (
            overall_strike_rate['total_runs'] / overall_strike_rate['balls_faced']
        ) * 100

        overall_strike_rate = overall_strike_rate[overall_strike_rate['total_runs'] > 1000]

        overall_strike_rate = overall_strike_rate.sort_values(
            by='strike_rate',
            ascending=False
        ).head(20)

        fig = px.bar(
            overall_strike_rate,
            x='strike_rate',
            y='batter',
            labels={
                'batter': 'Player',
                'strike_rate': 'Strike Rate'
            },
            color='batter',
            text='strike_rate',
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig.update_layout(
            height=600,
            width=800,
            yaxis_title='Player',
            xaxis_title='Strike Rate',
            yaxis=dict(categoryorder='total ascending')
        )

        fig.update_traces(textposition='outside')

        st.plotly_chart(
            fig,
            transparent=True,
            use_container_width=True
        )
        
    ########################################################################
    #######          Toss Decision Based On Top Venues         #############
    ########################################################################
    with st.expander('👉 Toss Decision Based on Top Venues'):
        top_venues = new_matchesDF['venue'].value_counts().head(
            15).index.to_list()
        top_venues_matches = new_matchesDF[new_matchesDF['venue'].isin(
            top_venues)]

        top_venues_matches['venue_location'] = top_venues_matches['venue'].astype(
            str) + ", " + top_venues_matches['city'].astype(str)

        venue_toss_stats = top_venues_matches.groupby(
            ['venue_location', 'toss_decision']
        ).size().reset_index(name='count')

        venue_toss_stats = venue_toss_stats.sort_values(
            by='count',
            ascending=False
        )

        fig = px.bar(
            venue_toss_stats,
            x='venue_location',
            y='count',
            color='toss_decision',
            labels={
                'count': 'Count',
                'venue_location': 'Venue and City'
            },
            color_discrete_sequence=px.colors.sequential.Viridis,
            barmode='group',
            text='count'
        )

        fig.update_layout(
            title_text='Toss Decision Based on Top Venues',
            xaxis_title='Venue and City',
            yaxis_title='Count',
            title_x=0.5,
            height=600,
            width=800,
            xaxis=dict(
                showgrid=False,
                tickangle=-45
            ),
            yaxis=dict(showgrid=True),
            plot_bgcolor='rgba(0,0,0,0)',
            legend_title_text='Decision Taken'
        )

        fig.update_traces(textposition='outside')

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        venue_toss_stats = venue_toss_stats.rename(
            columns={
                'count': 'Count',
                'venue_location': 'Venue and City',
                'toss_decision': 'Decision Taken'
            }
        )

        st.dataframe(
            venue_toss_stats.sort_values(
                by='Count',
                ascending=False
            ),
            width=800,
            height=400
        )

    #############################################################################
    ###########         Average Runs By Teams In Last Over         ##############
    #############################################################################
    with st.expander('👉 Average Runs Scored By Teams In Last Over'):
        last_over = new_deliveriesDF[new_deliveriesDF['over'] == 19]

        twenty_over_scores = last_over.groupby(
            'batting_team')['total_runs'].sum().sort_values(ascending=False)

        fig = px.bar(
            x=twenty_over_scores.values,
            y=twenty_over_scores.index,
            orientation='h',
            labels={
                'x': 'Total Runs',
                'y': 'Teams'
            },
            color=twenty_over_scores.index,
            text=twenty_over_scores.values,
            color_discrete_sequence=px.colors.sequential.Viridis
        )

        fig.update_traces(textposition='outside')
        fig.update_layout(
            yaxis=dict(
                autorange="reversed",
                showgrid=False
            ),
            xaxis=dict(
                showgrid=False
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            title='Total Runs Scored By Teams in Last Over Since 2008'
        )

        st.plotly_chart(fig, use_container_width=True)

    ##############################################################################
    #####           Total Runs Scored By Teams In Last Over Since 2008       #####
    ##############################################################################
    with st.expander('👉 Total Runs Scored By Teams In Last Over Since 2008'):
        last_over = new_deliveriesDF[new_deliveriesDF['over'] == 19]

        twenty_over_scores = last_over.groupby('batting_team')[
            'total_runs'].sum().sort_values(ascending=False)

        fig = px.bar(
            x=twenty_over_scores.values,
            y=twenty_over_scores.index,
            orientation='h',
            labels={
                'x': 'Total Runs',
                'y': 'Teams'
            },
            color=twenty_over_scores.index,
            text=twenty_over_scores.values,
            color_discrete_sequence=px.colors.sequential.Viridis,
        )

        plt.xticks(
            rotation=45,
            ha='right'
        )

        fig.update_traces(textposition='outside')

        fig.update_layout(
            yaxis=dict(
                autorange="reversed",
                showgrid=False
            ),
            xaxis=dict(showgrid=False),
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    ##############################################################################
    ###                    Total Runs Scored in Each Season                 ######
    ##############################################################################
    with st.expander('👉 Total Runs Scored in Each Season'):

        new_matchesDF['season'] = new_matchesDF['season'].astype(str)

        season_runs = new_matchesDF.groupby(
            'season')['target_runs'].sum().reset_index()

        temp4 = season_runs.set_index('season')

        fig = px.line(
            data_frame=season_runs,
            x='season',
            y='target_runs',
            labels={
                'target_runs': 'Total Runs',
                'season': 'Season'
            },
            markers=True,
            text='target_runs'
        )

        fig.update_layout(
            title='Total Runs Per Season',
            title_x=0.5,
            xaxis_title='Season',
            yaxis_title='Total Runs',
            xaxis=dict(
                showgrid=False,
                tickangle=-45
            ),
            yaxis=dict(showgrid=True),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        fig.update_traces(
            textposition='top center',
            marker=dict(size=8),
            line=dict(width=2)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    #####################################################################
    ###           Count of Matches By Different Umpires               ###
    #####################################################################
    with st.expander('👉 Count of Matches Umpired By Different Umpires'):
        umpires = pd.concat(
            [new_matchesDF['umpire1'], new_matchesDF['umpire2']]).value_counts()

        top_10_umpires = umpires.nlargest(10)

        fig = px.bar(
            x=top_10_umpires.values,
            y=top_10_umpires.index,
            orientation='h',
            labels={
                'x': 'Matches Umpired',
                'y': 'Umpire Name'
            },
            color=top_10_umpires.index,
            text=top_10_umpires.values,
            color_discrete_sequence=px.colors.sequential.Viridis
        )

        fig.update_traces(textposition='outside')
        fig.update_layout(
            yaxis=dict(
                autorange="reversed",
                showgrid=False
            ),
            xaxis=dict(
                showgrid=False
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            title='Top 10 Umpires Based on Matches Umpired'
        )

        st.plotly_chart(fig, use_container_width=True)

    #####################################################################
    ####              Teams with more than 200+ scores               ####
    #####################################################################
    with st.expander('👉 Teams With More Than 200+ Scores'):
        runs = new_deliveriesDF.groupby(
            ['match_id', 'inning', 'batting_team', 'bowling_team']
        )['total_runs'].sum().reset_index()

        runs_over_200_df = runs[runs['total_runs'] > 200]

        runs_over_200 = runs_over_200_df['batting_team'].value_counts()

        fig = px.bar(
            x=runs_over_200.values,
            y=runs_over_200.index,
            orientation='h',
            labels={
                'x': 'Number of Instances',
                'y': 'Teams'
            },
            text=runs_over_200.values,
            color=runs_over_200.index,
            color_discrete_sequence=px.colors.sequential.Viridis
        )

        fig.update_traces(textposition='outside')
        fig.update_layout(
            yaxis=dict(
                autorange="reversed",
                showgrid=False
            ),
            xaxis=dict(
                showgrid=False
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            title='Most 200+ Runs Scored By Teams'
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    ###########################################################################
    ############ Lucky Venue For Teams ########################################
    ###########################################################################
    with st.expander('👉  Lucky Venue For Teams'):
        teams = new_matchesDF.team1.unique().tolist()

        new_matchesDF['venue'] = new_matchesDF['venue']

        for team in teams:
            fig = plt.figure(figsize=(10, 5))

            team_name = team

            lucky_venues = new_matchesDF[
                new_matchesDF['winner'] == team_name
            ]['venue'].value_counts().nlargest(10)

            num_venues = len(lucky_venues)

            explode = [0.1] * num_venues
            colors = [
                'turquoise',
                'lightblue',
                'lightgreen',
                'crimson',
                'magenta',
                'orange'
            ]
            colors = colors * (num_venues // len(colors) + 1)
            colors = colors[:num_venues]

            lucky_venues.plot(
                kind='pie',
                autopct='%1.1f%%',
                explode=explode,
                shadow=True,
                startangle=20,
                colors=colors
            )

            plt.title(f'Win Percentage at Different Venues for {team}')

            plt.ylabel('')

            st.pyplot(
                fig,
                transparent=True
            )

    st.image("Images/divider.png")

