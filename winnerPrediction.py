import pickle
import pandas as pd
import streamlit as st


def app():
    st.markdown(
        '''
            <h1 style='text-align:center; color: #700961;'><strong> 🎲 PREDICTING WIN PROBABILITY FOR A TEAM 🎲</strong></h1>
            <h3 style='text-align:center; color: #ff01e7;'><strong>🚀 IMPROVED MODEL WITH ENHANCED FEATURES 🚀</strong></h3>
            <hr style="border-top: 3px solid #700961;">
        ''',
        unsafe_allow_html=True
    )

    # Loading the Saved Model and Scaler
    try:
        model = pickle.load(open('win_predictor_improved.pkl', 'rb'))
        scaler = pickle.load(open('win_predictor_scaler.pkl', 'rb'))
    except FileNotFoundError:
        st.error("Improved win prediction model not found. Please run train_win_predictor_improved.py first.")
        return

    # Team names
    TEAMS = [
        'Chennai Super Kings',
        'Delhi Capitals',
        'Kings XI Punjab',
        'Kolkata Knight Riders',
        'Mumbai Indians',
        'Rajasthan Royals',
        'Royal Challengers Bangalore',
        'Sunrisers Hyderabad'
    ]

    # Batting Team & Bowling Team
    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox('Batting Team', TEAMS)
    with col2:
        bowling_team = st.selectbox('Bowling Team', TEAMS)

    if bowling_team == batting_team:
        st.error("Bowling and Batting Team Can't Be The Same")
    else:
        encoded_batting_team = [
            1 if batting_team == TEAM else 0 for TEAM in TEAMS
        ]
        encoded_bowling_team = [
            1 if bowling_team == TEAM else 0 for TEAM in TEAMS
        ]

        # Current Runs
        current_runs = st.number_input(
            'Enter Current Score of Batting Team',
            min_value=0,
            step=1
        )

        # Wickets Out
        wickets_left = st.number_input(
            'Enter Number of Wickets Left For Batting Team',
            min_value=0,
            max_value=10,
            step=1
        )
        wickets_out = int(10 - wickets_left)

        # Overs Spent
        over = st.number_input(
            'Current Over of The Match',
            min_value=0,
            max_value=20,
            step=1
        )

        # Runs In Last 5
        run_lst_5 = st.number_input(
            'How Many Runs Batting Team Has Scored In Last 5 Overs?',
            min_value=0,
            step=1
        )

        # Wickets In Last 5
        wicket_lst_5 = st.number_input(
            'Number of Wickets Taken By Bowling Team In The Last 5 Overs?',
            min_value=0,
            step=1
        )

        # Additional features for improved model
        col1, col2 = st.columns(2)
        
        with col1:
            run_lst_3 = st.number_input(
                'Runs in Last 3 Overs?',
                min_value=0,
                step=1
            )
            
            boundary_rate = st.number_input(
                'Boundary Rate (4s & 6s per over)',
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                value=1.0
            )
        
        with col2:
            wicket_lst_3 = st.number_input(
                'Wickets in Last 3 Overs?',
                min_value=0,
                step=1
            )

        # Calculate additional features
        balls_left = (20 - over) * 6
        run_rate = current_runs / over if over > 0 else 0
        recent_run_rate = run_lst_5 / 5 if over >= 5 else run_rate
        recent_wicket_rate = wicket_lst_5 / 5 if over >= 5 else 0
        overs_remaining = 20 - over
        balls_per_wicket = balls_left / wickets_left if wickets_left > 0 else balls_left

        # Prepare data for prediction (enhanced features)
        data = [
            int(current_runs),
            int(wickets_out),
            over,
            int(run_lst_5),
            int(wicket_lst_5),
            int(run_lst_3),
            int(wicket_lst_3),
            balls_left,
            wickets_left,
            run_rate,
            recent_run_rate,
            recent_wicket_rate,
            overs_remaining,
            balls_per_wicket,
            boundary_rate
        ]

        data.extend(encoded_batting_team)
        data.extend(encoded_bowling_team)

        st.write('---')

        # Display match state
        st.subheader("📊 Current Match State")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Score", f"{current_runs}")
        with col2:
            st.metric("Wickets Lost", f"{wickets_out}")
        with col3:
            st.metric("Overs", f"{over}")
        with col4:
            st.metric("Run Rate", f"{run_rate:.2f}")

        # Additional metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Recent Run Rate", f"{recent_run_rate:.2f}")
        with col2:
            st.metric("Balls Left", f"{balls_left}")
        with col3:
            st.metric("Overs Remaining", f"{overs_remaining}")

        # Generating Predictions
        Generate_pred = st.button("🎯 Predict Win Probability")

        if Generate_pred:
            # Scale the data
            data_scaled = scaler.transform([data])
            
            # Get prediction probability
            prob = model.predict_proba(data_scaled)[0]
            batting_win_prob = prob[1]  # Probability that batting team wins
            bowling_win_prob = prob[0]  # Probability that bowling team wins

            st.write('---')
            st.subheader("🏆 Enhanced Win Probability Prediction")

            # Display probabilities with better styling
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    f"🏏 {batting_team}",
                    f"{batting_win_prob:.1%}",
                    delta=f"{batting_win_prob - 0.5:.1%}" if batting_win_prob > 0.5 else f"{batting_win_prob - 0.5:.1%}"
                )
                
            with col2:
                st.metric(
                    f"🏏 {bowling_team}",
                    f"{bowling_win_prob:.1%}",
                    delta=f"{bowling_win_prob - 0.5:.1%}" if bowling_win_prob > 0.5 else f"{bowling_win_prob - 0.5:.1%}"
                )

            # Prediction result with confidence
            confidence = max(batting_win_prob, bowling_win_prob)
            winning_team = batting_team if batting_win_prob > bowling_win_prob else bowling_team
            
            if confidence >= 0.8:
                st.success(f"🎯 **High Confidence Prediction: {winning_team} is likely to win!** (Confidence: {confidence:.1%})")
            elif confidence >= 0.6:
                st.warning(f"🎯 **Medium Confidence Prediction: {winning_team} is likely to win!** (Confidence: {confidence:.1%})")
            else:
                st.info(f"🎯 **Low Confidence Prediction: {winning_team} is likely to win!** (Confidence: {confidence:.1%})")

            # Additional insights
            st.write('---')
            st.subheader("📈 Match Insights")
            
            col1, col2 = st.columns(2)
            with col1:
                if run_rate > 8:
                    st.info("🔥 **High scoring rate detected!**")
                elif run_rate < 6:
                    st.info("🐌 **Low scoring rate detected!**")
                else:
                    st.info("⚖️ **Balanced scoring rate**")
                    
            with col2:
                if wickets_out >= 7:
                    st.info("⚠️ **Batting team under pressure!**")
                elif wickets_out <= 2:
                    st.info("💪 **Batting team in strong position!**")
                else:
                    st.info("🔄 **Moderate wicket loss**")
