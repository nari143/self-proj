import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.offline as pyo
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scrollToTop import create_scroll_to_top_button
from datasetPreprocessing import new_matchesDF, new_deliveriesDF


def app():
    st.markdown(
        '''
            <h1 style='text-align:center; color: #4ef037;'><strong> 🏏 TEAM ANALYSIS 🤾‍♂️ </strong></h1>
            <hr style="border-top: 3px solid #4ef037;">
        ''',
        unsafe_allow_html=True
    )

    Teams = new_matchesDF.team1.unique().tolist()
    team = st.selectbox("Select A Team", Teams)
    Analyze = st.button('Analyze')

    if Analyze:
        st.markdown(
            f"<h4 style='text-align: center;'> {team} </h4>",
            unsafe_allow_html=True
        )

        ###########################################################
        # -------->   AVERAGE SCORE AGAINST EACH TEAM      <-------
        ###########################################################
        selected_team_df = new_deliveriesDF[
            new_deliveriesDF['batting_team'].str.strip() == team
        ]

        innings_data = selected_team_df.groupby(
            [
                'match_id',
                'inning',
                'bowling_team'
            ]
        )['total_runs'].sum().reset_index()

        innings_data_scores = innings_data.groupby('bowling_team')['total_runs'].mean().round().astype(int).reset_index().sort_values(
            by='total_runs',
            ascending=False
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=innings_data_scores['bowling_team'],
                y=innings_data_scores['total_runs'],
                mode='lines+markers+text',
                text=innings_data_scores['total_runs'],
                textposition='top center',
                marker=dict(
                    color='purple',
                    size=10
                ),
                line=dict(
                    color='purple',
                    width=2
                ),
            )
        )

        for i in range(len(innings_data_scores)):
            fig.add_trace(
                go.Scatter(
                    x=[
                        innings_data_scores['bowling_team'][i],
                        innings_data_scores['bowling_team'][i]
                    ],
                    y=[
                        0,
                        innings_data_scores['total_runs'][i]
                    ],
                    mode='lines',
                    line=dict(
                        color='purple',
                        width=1,
                        dash='dot'
                    ),
                    showlegend=False
                )
            )

        fig.update_layout(
            title=f'Average Runs Scored By {team.strip()} Against Different Teams',
            xaxis_title="Bowling Teams",
            yaxis_title="Total Runs",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                tickmode='linear',
                fixedrange=True
            ),
            height=600,
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.image("Images/divider.png")

        ###########################################################
        # -> AVERAGE RUNS SCORED BY THE TEAM IN DIFFERENT OVERS   <-
        ###########################################################
        team_over_data = (
            selected_team_df.groupby('over')['total_runs'].mean()*6
        ).round().astype(int).reset_index()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=team_over_data['over'],
                y=team_over_data['total_runs'],
                mode='lines+markers+text',
                text=team_over_data['total_runs'],
                textposition='top center',
                marker=dict(
                    color='purple',
                    size=8
                ),
                line=dict(
                    color='purple',
                    width=2
                )
            )
        )

        for i in range(len(team_over_data)):
            fig.add_trace(
                go.Scatter(
                    x=[
                        team_over_data['over'][i],
                        team_over_data['over'][i]
                    ],
                    y=[
                        0,
                        team_over_data['total_runs'][i]
                    ],
                    mode='lines',
                    line=dict(
                        color='purple',
                        width=1,
                        dash='dot'
                    ),
                    showlegend=False
                )
            )

        fig.update_layout(
            title=f'Average Runs Scored By {team} in Different Overs',
            xaxis_title="Overs",
            yaxis_title="Total Runs",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                tickmode='linear',
                tick0=1,
                dtick=1,
                range=[0, 20],
                fixedrange=True
            ),
            height=600,
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.image("Images/divider.png")


######################################################################################################################################

        team_toss_decision = new_matchesDF[
            new_matchesDF['toss_winner'] == team
        ]['toss_decision']

        team_toss_decision_counts = team_toss_decision.value_counts().reset_index()

        team_toss_decision_counts.columns = ['toss_decision', 'count']

        fig = px.bar(
            team_toss_decision_counts,
            x='toss_decision',
            y='count',
            text='count',
            labels={
                'toss_decision': 'Toss Decision',
                'count': 'Count'
            },
            color_discrete_sequence=['purple'],
        )

        # Update layout for better aesthetics
        fig.update_layout(
            title=f'{team} Toss Decision',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                fixedrange=True
            ),
            yaxis=dict(
                fixedrange=True
            ),
            height=400,
        )

        # Show the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        st.image("Images/divider.png")


######################################################################################################################################

        st.markdown(
            f"<h5 style='text-align: center; color: white;'> {team} Match Wins Based On Venue </h5>",
            unsafe_allow_html=True
        )

        # Most Win Based On Venue,City
        venue_win = new_matchesDF[
            new_matchesDF['winner'] == team
        ]['venue'].value_counts()[:10]

        fig = plt.figure(
            figsize=(20, 5)
        )
        fig.patch.set_facecolor('white')
        ax = sns.barplot(
            x=venue_win.index,
            y=venue_win.values,
            color='black'
        )
        ax.set_facecolor('white')
        ax.bar_label(ax.containers[0])
        plt.xlabel('Venues',color='black')
        plt.ylabel('Wins',color='black')
        plt.xticks(
            fontsize=12,
            rotation='vertical',
            color='blue'
        )
        plt.yticks(
            fontsize=12,
            color='blue'
        )
        st.pyplot(
            fig,
            transparent=True,
        )

        st.image("Images/divider.png")

        st.markdown(
            f"<h5 style='text-align: center; color: white;'> {team} 200+ Runs </h5>",
            unsafe_allow_html=True
        )

        # Top 10 Highest Runs
        fig = plt.figure(
            figsize=(12, 10)
        )

        team_runs_over_200_df = selected_team_df.groupby(
            [
                'match_id',
                'bowling_team',
                'inning'
            ]
        )['total_runs'].sum().reset_index().sort_values(
            by='total_runs',
            ascending=False
        )

        team_runs_over_200 = team_runs_over_200_df[
            team_runs_over_200_df['total_runs'] > 200
        ]

        team_runs_over_200['data'] = team_runs_over_200['match_id'].astype(
            str) + "_" + team_runs_over_200['bowling_team']

        ax = sns.barplot(
            data=team_runs_over_200,
            y='data',
            x='total_runs'
        )

        ax.bar_label(ax.containers[0],color='white')
        plt.title(f'{team} 200+ Runs : Total({len(team_runs_over_200)})',color='white')
        plt.xlabel('Runs',color='white')
        plt.ylabel('Opponents',color='white')

        plt.xticks(
            fontsize=12,
            rotation='vertical',
            color='white'

        )
        plt.yticks(
            fontsize=12,
            color='white'
        )
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        st.pyplot(
            fig,
            transparent=True
        )

        st.image("Images/divider.png")
        st.markdown(
            f"<h5 style='text-align: center; color: white;'> {team} Top 10 Lowest Runs </h5>",
            unsafe_allow_html=True
        )

        # Top 10 Lowest Runs
        fig = plt.figure(
            figsize=(12, 10)
        )

        team_runs_over_df = selected_team_df.groupby(
            [
                'match_id',
                'bowling_team',
                'inning'
            ]
        )['total_runs'].sum().reset_index().sort_values(
            by='total_runs',
            ascending=True
        )

        team_runs_over_df = team_runs_over_df[
            team_runs_over_df['inning'] < 3
        ]

        team_runs_over_df = team_runs_over_df[:10]

        team_runs_over_df['data'] = team_runs_over_df['match_id'].astype(
            str)+"_"+team_runs_over_df['bowling_team']

        ax = sns.barplot(
            data=team_runs_over_df,
            y='data',
            x='total_runs'
        )

        ax.bar_label(ax.containers[0],color='white')
        plt.title(f'{team} Top 10 Lowest Runs',color='white')
        plt.xlabel('Runs',color='white')
        plt.ylabel('Opponents',color= 'white')

        plt.xticks(
            fontsize=12,
            rotation='vertical',
            color='white'
        )
        plt.yticks(
            fontsize=12,
            color='white'
        )
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        st.pyplot(
            fig,
            transparent=True
        )

        st.image("Images/divider.png")
    ### no of sixes and fours 
        st.markdown(
            f"<h5 style='text-align: center; color: white;'> {team} Fours and Sixes </h5>",
            unsafe_allow_html=True
        )

        fours_by_team = new_deliveriesDF[new_deliveriesDF['batsman_runs'] == 4].groupby('batting_team')['batsman_runs'].count().reset_index()
        fours_by_team.columns = ['Team', 'Fours']

        # ✅ SIXES (batsman_runs == 6)
        sixes_by_team = new_deliveriesDF[new_deliveriesDF['batsman_runs'] == 6].groupby('batting_team')['batsman_runs'].count().reset_index()
        sixes_by_team.columns = ['Team', 'Sixes']

        # Merge the two DataFrames on 'Team'
        fours_sixes_by_team = pd.merge(fours_by_team, sixes_by_team, on='Team')
        # Set the index to 'Team' for better plotting
        fours_sixes_by_team.set_index('Team', inplace=True)
        # Plotting the Fours and Sixes
        fig = plt.figure(figsize=(12, 6))
        fig.patch.set_facecolor('black')
        ax = sns.barplot(data=fours_sixes_by_team, palette='viridis')
       # Label the first set (Fours)
        ax.bar_label(
            ax.containers[0],
            labels=[f"{int(bar.get_height())}" for bar in ax.containers[0]],
            color='white'
        )

        # Label the second set (Sixes)
        ax.bar_label(
            ax.containers[1],
            labels=[f"{int(bar.get_height())}" for bar in ax.containers[1]],
            color='white'
        )
        plt.title(f'{team} Fours and Sixes', color='white')
        plt.xlabel('Teams', color='white')
        plt.ylabel('Count', color='white')
        plt.xticks(
            fontsize=12,
            rotation='vertical',
            color='white'
        )
        plt.yticks(
            fontsize=12,
            color='white'
        )
        ax.set_facecolor('black')
        st.pyplot(
            fig,
            transparent=True
        )
        
    create_scroll_to_top_button(key_suffix="teamAnalysis")
    st.image("Images/divider.png")
