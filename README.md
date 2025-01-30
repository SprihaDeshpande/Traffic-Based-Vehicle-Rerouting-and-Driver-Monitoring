# Traffic-Based-Vehicle-Rerouting-and-Driver-Monitoring
A Comprehensive Framework for Traffic-Based Vehicle Rerouting and Driver Monitoring 

🚀 Overview
Driver fatigue is a major cause of road accidents, making real-time monitoring essential for transportation safety. This project presents an intelligent driver alertness detection system that integrates:

Machine Learning-based Eye Status Prediction using CNNs trained on the MRL Eye Dataset
Google Maps API for Dynamic Rerouting based on real-time traffic data
Streamlit for an Interactive Web Interface
By combining deep learning-based fatigue detection with intelligent route planning, this system enhances road safety and traffic management.

🔑 Features
✅ Real-time Driver Monitoring via image-based classification
✅ Fatigue Detection using CNNs trained on the MRL Eye Dataset
✅ Dynamic Route Planning with Google Maps API integration
✅ Streamlit-powered Web App for an intuitive user experience
✅ Seamless Traffic-aware Rerouting to avoid congestion
✅ Alerts for Drowsy Driving for enhanced road safety

🖥️ Tech Stack
Machine Learning: TensorFlow, OpenCV, CNN
Deep Learning Dataset: MRL Eye Dataset
API Integration: Google Maps API
Web Framework: Streamlit
Languages: Python
📂 Project Structure
graphql
Copy
Edit
📦 driver-alertness-detection  
│-- 📂 models/            # Trained CNN model  
│-- 📂 dataset/           # Preprocessed MRL Eye Dataset  
│-- 📂 static/            # Images, icons, or assets  
│-- 📂 src/               # Main application code  
│   │-- app.py            # Streamlit UI  
│   │-- detection.py      # Eye status classification  
│   │-- routing.py        # Traffic-based rerouting logic  
│-- requirements.txt      # Python dependencies  
│-- README.md             # Project documentation  
🔧 Setup & Installation
1️⃣ Clone the repository

bash
Copy
Edit
git clone https://github.com/your-username/driver-alertness-detection.git
cd driver-alertness-detection
2️⃣ Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Set up Google Maps API

Get your API key from Google Cloud Console
Store it in an .env file:
makefile
Copy
Edit
GOOGLE_MAPS_API_KEY=your_api_key_here
4️⃣ Run the application

bash
Copy
Edit
streamlit run src/app.py
📊 Model Performance
The CNN model was trained on the MRL Eye Dataset and achieved:

Accuracy: 95.3%
Precision: 94.1%
Recall: 96.7%
📌 Future Enhancements
🔹 Improve model accuracy with larger datasets
🔹 Integrate real-time video processing for live monitoring
🔹 Expand rerouting logic with AI-based traffic prediction

🤝 Contributions
Want to improve this project? Feel free to fork, submit issues, or open a PR!

📜 References
📄 [1] Google Maps API - Documentation
📄 [2] Streamlit - Official Website
📄 [3] MRL Eye Dataset - Kaggle

