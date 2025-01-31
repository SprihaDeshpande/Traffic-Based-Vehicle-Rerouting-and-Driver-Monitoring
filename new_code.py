import streamlit as st
import time
import streamlit.components.v1 as components
from PIL import Image
from model import DriverEyeStatusModel
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tempfile
from tensorflow.keras.models import load_model
model = load_model('eye_status_model.h5')

# Google Maps API Key (replace with your actual API Key)
google_maps_api_key = "Google_API_KEY"  # Replace with your actual API key

def create_map_html(origin, destination, api_key):
    return f"""
        <html>
        <head>
        <script src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap" async defer></script>
        <style>
            /* Ensure the map div fills the screen */
            html, body {{
                height: 100%;   /* Full height of the page */
                margin: 0;
                padding: 0;
            }}
            #map {{
                height: 100%;   /* Map div will take up the full height of the page */
                width: 100%;    /* Map div will take up the full width of the page */
            }}
        </style>

        <script type="text/javascript">
            var map;
            var directionsService;
            var directionsRenderer1, directionsRenderer2;
            var carMarker1, carMarker2;
            var trafficPointReached = false;
            var trafficCoordinates = [];
            var car2Rerouted = false;

            function initMap() {{
                directionsService = new google.maps.DirectionsService();
                directionsRenderer1 = new google.maps.DirectionsRenderer();
                directionsRenderer2 = new google.maps.DirectionsRenderer();
                map = new google.maps.Map(document.getElementById("map"), {{
                    zoom: 14,
                    center: {{lat: 37.773972, lng: -122.431297}}  // Default center (adjust as necessary)
                }});
                directionsRenderer1.setMap(map);
                directionsRenderer2.setMap(map);
                directionsRenderer2.setOptions({{suppressMarkers: false}});  // Show Car 2's route with markers

                // Car 1 initial route
                var request1 = {{
                    origin: '{origin}',
                    destination: '{destination}',
                    travelMode: google.maps.TravelMode.DRIVING,
                    drivingOptions: {{
                        departureTime: new Date(),
                        trafficModel: google.maps.TrafficModel.PESSIMISTIC
                    }}
                }};

                directionsService.route(request1, function(result, status) {{
                    if (status == google.maps.DirectionsStatus.OK) {{
                        directionsRenderer1.setDirections(result);
                        var route = result.routes[0];
                        var legs = route.legs;
                        var steps = [];
                        for (var i = 0; i < legs.length; i++) {{
                            for (var j = 0; j < legs[i].steps.length; j++) {{
                                steps.push(legs[i].steps[j].end_location);
                            }}
                        }}
                        moveCarAlongRoute(steps);
                    }} else {{
                        console.error('Directions request failed due to ' + status);
                    }}
                }});

                // Car 2 initial route (hidden by default)
                var request2 = {{
                    origin: '{origin}',
                    destination: '{destination}',
                    travelMode: google.maps.TravelMode.DRIVING,
                    drivingOptions: {{
                        departureTime: new Date(),
                        trafficModel: google.maps.TrafficModel.BEST_GUESS
                    }},
                    avoidFerries: true,
                    avoidHighways: true
                }};

                directionsService.route(request2, function(result, status) {{
                    if (status == google.maps.DirectionsStatus.OK) {{
                        directionsRenderer2.setDirections(result);
                        var route = result.routes[0];
                        var legs = route.legs;
                        var steps = [];
                        for (var i = 0; i < legs.length; i++) {{
                            for (var j = 0; j < legs[i].steps.length; j++) {{
                                steps.push(legs[i].steps[j].end_location);
                            }}
                        }}
                        moveCar2AlongRoute(steps);
                    }} else {{
                        console.error('Directions request failed due to ' + status);
                    }}
                }});
            }}

            function moveCarAlongRoute(steps) {{
                var stepIndex = 0;
                carMarker1 = new google.maps.Marker({{
                    position: steps[0],
                    map: map,
                    icon: 'https://maps.google.com/mapfiles/ms/icons/cabs.png'  // Car 1 icon
                }});

                function moveCar() {{
                    if (stepIndex < steps.length) {{
                        carMarker1.setPosition(steps[stepIndex]);
                        stepIndex++;
                        if (stepIndex % 3 === 0 && !trafficCoordinates.includes(steps[stepIndex])) {{
                            console.log("Traffic encountered by Car 1 at step", stepIndex);
                            trafficCoordinates.push(steps[stepIndex]);
                            console.log("Sending message to Car 2: Traffic detected at coordinates", steps[stepIndex]);
                            sendMessageToCar2(steps[stepIndex]);
                        }}
                        setTimeout(moveCar, 2000);  // Move Car 1 every 2 seconds
                    }} else {{
                        console.log('Car 1 has reached the destination');
                    }}
                }}

                moveCar();
            }}

            function sendMessageToCar2(coordinates) {{
                alert("Message sent to Car 2: Reroute due to traffic at " + coordinates);
            }}

            function rerouteCar2() {{
                console.log("Car 2 rerouting...");
                for (var i = 0; i < trafficCoordinates.length; i++) {{
                    var trafficPoint = trafficCoordinates[i];
                    if (!car2Rerouted) {{
                        directionsRenderer2.setOptions({{suppressMarkers: false}});
                        var request2 = {{
                            origin: trafficPoint,
                            destination: '{destination}',
                            travelMode: google.maps.TravelMode.DRIVING,
                            drivingOptions: {{
                                departureTime: new Date(),
                                trafficModel: google.maps.TrafficModel.PESSIMISTIC
                            }},
                            avoidTolls: true
                        }};

                        directionsService.route(request2, function(result, status) {{
                            if (status == google.maps.DirectionsStatus.OK) {{
                                directionsRenderer2.setDirections(result);
                                var route = result.routes[0];
                                var legs = route.legs;
                                var steps = [];
                                for (var i = 0; i < legs.length; i++) {{
                                    for (var j = 0; j < legs[i].steps.length; j++) {{
                                        steps.push(legs[i].steps[j].end_location);
                                    }}
                                }}
                                moveCar2AlongRoute(steps);
                            }} else {{
                                console.error("Failed to reroute Car 2: ", status);
                            }}
                        }});
                        car2Rerouted = true;
                    }}
                }}
            }}

            function moveCar2AlongRoute(steps) {{
                var stepIndex = 0;
                carMarker2 = new google.maps.Marker({{
                    position: steps[0],
                    map: map,
                    icon: 'https://maps.google.com/mapfiles/ms/icons/truck.png'  // Car 2 icon
                }});

                function moveCar2() {{
                    if (stepIndex < steps.length) {{
                        carMarker2.setPosition(steps[stepIndex]);
                        stepIndex++;
                        setTimeout(moveCar2, 2500);  // Move Car 2 every 2.5 seconds
                    }} else {{
                        console.log("Car 2 has reached the destination");
                    }}
                }}

                moveCar2();
            }}
        </script>
        </head>
        <body>
            <div id="map"></div>  <!-- Map div with no inline styles, will fill the page as defined in the CSS -->
        </body>
        </html>
    """

import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def check_driver_status(driver_image):
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(driver_image.getvalue())
        tmp_file_path = tmp_file.name  # Store the file path

    # Set the directory paths for training and validation (you might need to change these paths)
    train_dir = '/Users/sprihad/Desktop/essay4/dataset/train'  
    validation_dir = '/Users/sprihad/Desktop/essay4/dataset/validation'
    
    # Initialize the model
    model = DriverEyeStatusModel(train_dir, validation_dir)  # Create an instance of the model

    # Train the model (if not already trained)
    # model.train_model(epochs=20)  # Optional, use if you want to train it every time

    # Load the trained model (this is your pre-trained model)
    model = load_model('eye_status_model.h5')  # Load the model from the saved file

    # Load and preprocess the image
    img = load_img(tmp_file_path, target_size=(224, 224))  # Resize image to match the model's input size
    img_array = img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image to [0, 1]

    # Predict driver status using the image array
    driver_status_prob = model.predict(img_array)  # Predict the status

    # Interpret the result (assuming 0 = sleeping, 1 = awake)
    driver_status = "Drive Awake" if driver_status_prob >= 0.5 else "Driver Sleeping"

    # Based on the prediction result, you can take actions
    if driver_status == "Drive Awake":
        print("Driver is awake, continuing drive")
    elif driver_status == "Driver Sleeping":
        print("Driver sleepy, pulling to shoulder")

    # Example condition simulation (this can be replaced with actual detection logic)
    driver_alert = "Driver Alert: Eyes Open"
    driver_sleepy = "Driver Sleepy: Eyes Closed"

    # Simulating: Assuming eyes are closed (replace with real logic)
    driver_status = driver_sleepy if "eyes_closed" in tmp_file_path else driver_alert

    return driver_status



# Streamlit App Layout
st.title("Real-Time Car and Truck Simulation with Traffic-Based Rerouting")

# User input for origin and destination
origin = st.text_input("Enter Origin (e.g., 'San Francisco, CA')", "San Francisco, CA")
destination = st.text_input("Enter Destination (e.g., 'Los Angeles, CA')", "Los Angeles, CA")

# Upload driver image to simulate alertness detection
driver_image = st.file_uploader("Upload Driver's Image (Eyes Open/Closed)", type=["png", "jpg", "jpeg"])

if driver_image:
    # Check driver's status based on the uploaded image
    driver_status = check_driver_status(driver_image)
    st.image(driver_image, caption="Uploaded Driver Image", use_column_width=True)
    st.write(f"Driver Status: {driver_status}")
    
    # If driver is sleepy, notify Car 2 to pull over
    if "Sleepy" in driver_status:
        st.write("Alert: Car 2 is requested to park on the side as Driver is sleepy.")
        # Code for Car 2 to reroute and park on the side
        st.write("Car 2 rerouting to park on the side.")
        # Call Car 2 parking or rerouting logic here

# Display map for origin, destination and the car's route
components.html(create_map_html(origin, destination, google_maps_api_key))

import streamlit as st

import time
import random

# Function to calculate HRmax, HRR, THRR, and THRmax
def calculate_target_heart_rate(age, resting_heart_rate, intensity_range=(0.4, 0.59)):
    # Calculate HRmax using the age-based formula
    HRmax = 220 - age  # Simplified formula

    # Calculate HRR (Heart Rate Reserve)
    HRR = HRmax - resting_heart_rate

    # Calculate Target Heart Rate Range (THRR) for the given intensity range
    THRR_min = (HRR * intensity_range[0]) + resting_heart_rate
    THRR_max = (HRR * intensity_range[1]) + resting_heart_rate

    # Calculate THRmax (Target Heart Rate Max) for the desired intensity
    THRmax_min = HRmax * intensity_range[0]
    THRmax_max = HRmax * intensity_range[1]

    return HRmax, HRR, THRR_min, THRR_max, THRmax_min, THRmax_max

# Streamlit UI elements
st.title("Real-Time Heart Rate Monitoring")
st.write("This tool simulates real-time heart rate data and checks whether it falls within a healthy range based on the Karvonen formula.")

# User input for age and resting heart rate
age = st.number_input("Enter your age", min_value=18, max_value=100, value=30)
resting_heart_rate = st.number_input("Enter your resting heart rate (bpm)", min_value=40, max_value=100, value=72)

# Optional: Let the user adjust the intensity range for the calculation
min_intensity = st.slider("Select minimum intensity (%)", 30, 60, 40) / 100
max_intensity = st.slider("Select maximum intensity (%)", 60, 85, 59) / 100

# Calculate the target heart rate values
HRmax, HRR, THRR_min, THRR_max, THRmax_min, THRmax_max = calculate_target_heart_rate(
    age, resting_heart_rate, intensity_range=(min_intensity, max_intensity)
)

# Display the results
st.write(f"**Maximum Heart Rate (HRmax):** {HRmax} bpm")
st.write(f"**Heart Rate Reserve (HRR):** {HRR} bpm")
st.write(f"**Target Heart Rate Range (THRR):** {THRR_min:.1f} - {THRR_max:.1f} bpm")
st.write(f"**Target Heart Rate Max (THRmax):** {THRmax_min:.1f} - {THRmax_max:.1f} bpm")

# Initialize session state for monitoring
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

# Start and stop buttons
col1, col2 = st.columns(2)
start_button = col1.button("Start Monitoring")
stop_button = col2.button("Stop Monitoring")

# Logic to handle start and stop
if start_button:
    st.session_state.monitoring = True
if stop_button:
    st.session_state.monitoring = False

# Continuous monitoring
if st.session_state.monitoring:
    st.write("**Monitoring Heart Rate in Real-Time...**")
    monitoring_placeholder = st.empty()

    # Simulate continuous heart rate data
    for _ in range(100):  # Simulate 100 readings; adjust as needed
        if not st.session_state.monitoring:
            break  # Stop monitoring if "Stop Monitoring" is clicked
        simulated_heart_rate = random.randint(50, 150)  # Simulate heart rate in bpm
        if THRR_min <= simulated_heart_rate <= THRR_max:
            status = "✅ Healthy"
        elif simulated_heart_rate < THRR_min:
            status = "⚠️ Below Target"
        else:
            status = "⚠️ Above Target"

        monitoring_placeholder.markdown(
            f"""
            **Current Heart Rate:** {simulated_heart_rate} bpm  
            **Status:** {status}
            """
        )

        time.sleep(1)  # Update every second
else:
    st.write("**Monitoring is stopped. Click 'Start Monitoring' to begin.**")