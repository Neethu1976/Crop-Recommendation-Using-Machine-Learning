# Crop Recommendation System

A machine learning-powered web application that recommends the best crops based on soil and environmental conditions.

## Overview

This project is a Crop Recommendation System designed to help users determine the most suitable crops to plant based on specific soil and environmental parameters. It leverages a Random Forest classifier to analyze user-provided data and suggest crops with confidence scores, making it a practical tool for farmers, agronomists, and agricultural enthusiasts. The app provides detailed crop information, alternative recommendations, and insights into key influencing factors for informed decision-making.

## Features

- Personalized crop recommendations based on soil nutrients and environmental factors
- Considers key parameters: Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, soil pH, and rainfall
- User-friendly web interface with input forms and results display
- Displays confidence scores for predictions, along with crop descriptions, seasons, harvest times, and emojis
- Shows top 3 alternative crop suggestions with their confidence levels
- Highlights key factors influencing the recommendation using feature importance analysis
- Responsive design for desktop and mobile devices
- Input validation and error handling
- Logging for debugging and monitoring
- Easily extensible for additional crops and features

## Technologies Used

- Backend: Python, Flask
- Machine Learning: scikit-learn Random Forest Classifier
- Frontend: HTML5, CSS3, Jinja2 templating
- Data Processing: NumPy, Pandas
- Model Persistence: joblib

## Project Structure

```
├── app.py                  # Flask backend application with crop info and prediction logic
├── models/
│   └── RandomForest.pkl    # Trained RandomForest model
├── templates/
│   ├── index.html          # Input form for parameters
│   └── result.html         # Results display page with detailed crop info
├── static/
│   └── style.css           # Responsive styling for the application
├── project_pipeline.txt    # Detailed project development pipeline
├── README.md               # Project documentation
└── TODO.md                 # Completed tasks and notes
```

## Installation

1. Clone or download the project files to your local machine.

2. Install dependencies (requirements.txt is not included, so install manually):
   ```bash
   pip install flask scikit-learn numpy pandas joblib
   ```

3. Ensure the trained model file `RandomForest.pkl` is present in the `models/` directory.

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://localhost:5000` to access the web interface.

## Usage

### Web Interface

1. Fill out the form with your soil and environmental parameters:
   - Soil pH Level (0-14, optimal 6.0-7.5)
   - Nitrogen content (kg/ha, 0-150)
   - Phosphorus content (kg/ha, 0-150)
   - Potassium content (kg/ha, 0-250)
   - Temperature (°C, -10 to 50)
   - Humidity (%, 0-100)
   - Rainfall (mm, 0-300)

2. Click "Get Recommendations" to receive crop suggestions.

3. View the results page, which includes:
   - The recommended crop with its description, season, harvest time, and emoji.
   - Confidence score for the prediction.
   - Top 3 alternative crops with their confidence levels and emojis.
   - Key factors influencing the recommendation (based on feature importances), ranked by importance with visual bars.
   - A recap of your input values for reference.

## Supported Crops

The system can recommend from 22 varieties of crops, each with detailed information:

- Apple 🍎 - Temperate fruit, rich in fiber and antioxidants
- Banana 🍌 - Tropical fruit, high in potassium
- Blackgram 🫘 - Protein-rich pulse crop
- Chickpea 🫘 - Versatile legume high in protein
- Coconut 🥥 - Tropical palm fruit
- Coffee ☕ - Popular beverage crop
- Cotton 🌱 - Fiber crop for textiles
- Grapes 🍇 - Used for wine and eating
- Jute 🌾 - Fiber crop for textiles
- Kidney Beans 🫘 - Nutritious legumes
- Lentil 🫘 - Protein-rich pulses
- Maize 🌽 - Staple grain crop
- Mango 🥭 - Sweet tropical fruit
- Moth Beans 🫘 - Drought-resistant legumes
- Mungbean 🫘 - Fast-growing legumes
- Muskmelon 🍈 - Sweet melons
- Orange 🍊 - Citrus fruit rich in vitamin C
- Papaya 🍈 - Tropical fruit
- Pigeon Peas 🫘 - Drought-tolerant legumes
- Pomegranate 🍎 - Antioxidant-rich fruit
- Rice 🌾 - Staple cereal crop
- Watermelon 🍉 - Refreshing summer fruit

## Development

To modify or extend the system:

- Update or retrain the model using your own agricultural data.
- Customize the frontend by editing the HTML templates and CSS.
- Extend backend logic in `app.py` for additional features or API endpoints.
- Add more crops to the `crop_info` dictionary with descriptions, seasons, harvest times, and emojis.

### Project Pipeline

The development followed a comprehensive pipeline:

1. **Data Collection and Preprocessing**: Collected agricultural dataset, cleaned data, handled missing values.
2. **Exploratory Data Analysis (EDA)**: Analyzed distributions, correlations, and patterns.
3. **Model Selection and Training**: Chose Random Forest, split data, trained and evaluated model.
4. **Model Evaluation and Optimization**: Assessed metrics, performed hyperparameter tuning.
5. **Web Application Development**: Built Flask app, created templates, integrated model.
6. **Feature Enhancements**: Added confidence scoring, crop details, alternatives, feature importance.
7. **Deployment and Testing**: Tested locally, ensured responsiveness.

### Recent Enhancements

- Added comprehensive crop information dictionary with descriptions, seasons, harvest times, and emojis for all 22 crops.
- Enhanced prediction function to provide detailed crop data, alternatives, and key factors.
- Updated results template to display crop descriptions, alternative recommendations, and feature importance insights.
- Improved UI/UX with modern design, responsive layout, and interactive elements.
- Implemented input validation, error handling, and logging for better reliability.

### Future Improvements

- Integrate real-time weather data for more accurate recommendations.
- Add user feedback mechanism to improve model over time.
- Expand crop database with more varieties and regional data.
- Implement user accounts for personalized recommendations and history tracking.
- Add API endpoints for integration with other agricultural tools.

## License

This project is open source and available under the MIT License.

---

**Note:** This is a demonstration system. For production use, train the model with real agricultural data and consider additional factors like local climate patterns, crop rotation, and market conditions.
