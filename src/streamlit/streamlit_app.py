import streamlit as st
from PIL import Image
from src.features.ms3_clean import print_data
from datetime import datetime
from src.client.client import *
from src.client.gameclient import *

gc = GameClient(ip="serving")
sclient = ServingClient(ip="serving")

# Displaying the NHL Logo
col1, col2, col3 = st.columns(3)

with col1:
    st.write(" ")

with col2:
    img = Image.open("src/streamlit/Nhl_logo.png")
    img = img.resize((img.width // 6, img.height // 6))
    st.image(img)

with col3:
    st.write(" ")

# Title of the App
st.markdown(
    "<h1 style='text-align: center; color: black;'>Hockey Visualisation App</h1>",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.sidebar.title("Model selection")

    workspace = st.text_input("Workspace", "A11-Group")
    side_model = st.sidebar.selectbox(
        "Model", ("LogReg Distance", "LogReg Distance & Angle")
    )
    version = st.text_input("Version", value="1.0.0")

    get_model = st.button("Get Model")

    if side_model == "LogReg Distance & Angle":
        model = "logit_dist_angle"
    elif side_model == "LogReg Distance":
        model = "logit_distance"

    if get_model:
        response = sclient.download_registry_model(
            model=model, workspace="tedoul", version="1.0.0"
        )

        response = response["status"]

        st.markdown(
            f"<h4 style='text-align: center; color: green;'>{response}</h4>",
            unsafe_allow_html=True,
        )

user_input = st.text_input("GameID")

button = st.button("Ping game")


if button:
    gd = gc.fetch_live_game_data(user_input)

    if gd == "vide":
        st.markdown(
            f"<h2 style='text-align: center; color: black; background-color: lightcoral;'>Ce GameId n'existe pas</h2>",
            unsafe_allow_html=True,
        )
    else:
        away_team = gd["awayTeam"]["name"]["default"]
        home_team = gd["homeTeam"]["name"]["default"]

        year = int(user_input[0:4])
        gametype = user_input[4:6]
        if gametype == "01":
            g_type = "Regular season ##\n"
        elif gametype == "02":
            g_type = "All-Star ##\n"
        elif gametype == "03":
            g_type = "Playoffs ##\n"

        saisons = f"""
                ## Saison {year}/{year+1} 
                ## {g_type} """

        teams = f"{home_team} vs {away_team}"

        st.markdown(
            f"<h1 style='text-align: center; color: black;'>{saisons}</h1>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<h3 style='text-align: center; color: black; background-color: aliceblue;'>{teams}</h3>",
            unsafe_allow_html=True,
        )

    data = print_data(gd)

    st.write(" ")
    st.write(" ")

    # st.slider("Event", len(data))

    st.markdown(
        f"<h5 style='text-align: center; color: black;'>{home_team} xG (actual)&emsp;&emsp;&emsp;&emsp;&emsp; {away_team} xG (actual)</h5>",
        unsafe_allow_html=True,
    )

    away_score = gd["awayTeam"]["score"]
    home_score = gd["homeTeam"]["score"]

    home_team_score = 0
    away_team_score = 0

    for index, row in data.iterrows():
        if row["is_goal"] == 1:
            if row["event_owner_team_id"] == row["home_team_id"]:
                home_team_score += 1
            else:
                away_team_score += 1
        data.at[index, "home_team_goals"] = home_team_score
        data.at[index, "away_team_goals"] = away_team_score

        if row["period"] in [1, 2, 3]:
            time_in_period = datetime.strptime("20:00", "%M:%S")
            time_played_in_period = datetime.strptime(row["period_time"], "%M:%S")

            time_left = time_in_period - time_played_in_period
            minutes, seconds = divmod(time_left.seconds, 60)
            formatted_time_left = f"{minutes:02d}:{seconds:02d}"

            data.at[index, "time_left_in_period"] = formatted_time_left
        else:
            time_in_period = datetime.strptime("05:00", "%M:%S")
            time_played_in_period = datetime.strptime(row["period_time"], "%M:%S")

            time_left = time_in_period - time_played_in_period
            minutes, seconds = divmod(time_left.seconds, 60)
            formatted_time_left = f"{minutes:02d}:{seconds:02d}"

            data.at[index, "time_left_in_period"] = formatted_time_left

    features = [
        "shot_type",
        "x_coordinate",
        "y_coordinate",
        "away_team_players",
        "home_team_players",
        "empty_net",
        "distance_to_net",
        "shot_angle",
        "home_team_goals",
        "away_team_goals",
        "time_left_in_period",
        "prediction",
    ]

    df_shot = data[data["event_type"] == "shot-on-goal"]
    df_shot = df_shot.reset_index(drop=True)

    if side_model == "LogReg Distance & Angle":
        columns = ["distance_to_net", "shot_angle"]
    elif side_model == "LogReg Distance":
        columns = ["distance_to_net"]

    data = df_shot[columns].values
    df = pd.DataFrame(data, columns=columns)
    predictions = sclient.predict(df)

    df_shot = pd.concat([df_shot, predictions], axis=1)

    home_team_cum_prob = 0
    away_team_cum_prob = 0

    for index, row in df_shot.iterrows():
        if row["home_team_id"] == row["event_owner_team_id"]:
            home_team_cum_prob += row["prediction"]
        else:
            away_team_cum_prob += row["prediction"]

    difference_home = round(home_score - home_team_cum_prob, 2)
    difference_away = round(away_score - away_team_cum_prob, 2)

    if difference_away < 0:
        fleche_away = "↓"
    else:
        fleche_away = "↑"

    if difference_home < 0:
        fleche_home = "↓"
    else:
        fleche_home = "↑"

    st.markdown(
        f"<h3 style='text-align: center; color: black;'>{round(home_team_cum_prob, 2)} ({home_score})&emsp;&emsp;&emsp;&emsp;&emsp; {round(away_team_cum_prob, 2)} ({away_score})</h3>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<h5 style='text-align: center; color: blue;'>{fleche_home} {difference_home}&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; {fleche_away} {difference_away}</h5>",
        unsafe_allow_html=True,
    )

    st.write(df_shot[features])
