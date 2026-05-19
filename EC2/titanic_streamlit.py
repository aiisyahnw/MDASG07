import json
import boto3
import streamlit as st


ENDPOINT_NAME = "spaceship-endpoint-v4"
REGION = "us-east-1"

runtime = boto3.client(
    "sagemaker-runtime",
    region_name=REGION
)


st.set_page_config(
    page_title="Spaceship Titanic Deployment"
)

st.title("Spaceship Titanic Model Deployment")

st.write("Enter passenger data")


col1, col2 = st.columns(2)

with col1:

    home_planet = st.selectbox(
        "Home Planet",
        ["Earth", "Europa", "Mars"]
    )

    cryo_sleep = st.selectbox(
        "CryoSleep",
        [True, False]
    )

    destination = st.selectbox(
        "Destination",
        [
            "TRAPPIST-1e",
            "PSO J318.5-22",
            "55 Cancri e"
        ]
    )

    age = st.slider(
        "Age",
        0,
        100,
        25
    )

    vip = st.selectbox(
        "VIP",
        [False, True]
    )


with col2:

    room_service = st.number_input(
        "Room Service",
        0.0,
        10000.0,
        0.0
    )

    food_court = st.number_input(
        "Food Court",
        0.0,
        10000.0,
        0.0
    )

    shopping_mall = st.number_input(
        "Shopping Mall",
        0.0,
        10000.0,
        0.0
    )

    spa = st.number_input(
        "Spa",
        0.0,
        10000.0,
        0.0
    )

    vr_deck = st.number_input(
        "VR Deck",
        0.0,
        10000.0,
        0.0
    )


if st.button("Predict Result"):

    features = {
        "Age": age,
        "RoomService": room_service,
        "FoodCourt": food_court,
        "ShoppingMall": shopping_mall,
        "Spa": spa,
        "VRDeck": vr_deck,
        "HomePlanet": home_planet,
        "CryoSleep": cryo_sleep,
        "Destination": destination,
        "VIP": vip
    }

    try:

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(features)
        )

        result = json.loads(
            response["Body"]
            .read()
            .decode()
        )

        st.divider()

        if result["prediction"] == "Transported":

            st.success("RESULT: TRANSPORTED")
            st.balloons()

        else:

            st.error("RESULT: NOT TRANSPORTED")

    except Exception as e:

        st.error(f"Prediction Error: {e}")