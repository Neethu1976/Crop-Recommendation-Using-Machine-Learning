import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Feature names and crop classes
feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
crop_classes = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton',
    'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans',
    'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate',
    'rice', 'watermelon'
]

# Enhanced crop descriptions with growing tips
crop_info = {
    'apple': {
        'description': 'Apples are nutritious fruits rich in fiber and antioxidants. They thrive in temperate climates with well-drained soil.',
        'season': 'Fall',
        'harvest_time': '120-180 days',
        'emoji': '🍎'
    },
    'banana': {
        'description': 'Bananas are tropical fruits high in potassium. They require warm, humid conditions and regular watering.',
        'season': 'Year-round',
        'harvest_time': '9-12 months',
        'emoji': '🍌'
    },
    'blackgram': {
        'description': 'Blackgram is a protein-rich pulse crop. It grows well in warm climates with moderate rainfall.',
        'season': 'Kharif',
        'harvest_time': '80-90 days',
        'emoji': '🫘'
    },
    'chickpea': {
        'description': 'Chickpeas are versatile legumes high in protein. They prefer cool, dry conditions during growth.',
        'season': 'Rabi',
        'harvest_time': '90-100 days',
        'emoji': '🫘'
    },
    'coconut': {
        'description': 'Coconuts are tropical palm fruits. They need sandy, well-drained soil and abundant sunlight.',
        'season': 'Year-round',
        'harvest_time': '12 months',
        'emoji': '🥥'
    },
    'coffee': {
        'description': 'Coffee is a popular beverage crop. It grows best in high-altitude, tropical regions with rich soil.',
        'season': 'Year-round',
        'harvest_time': '3-4 years',
        'emoji': '☕'
    },
    'cotton': {
        'description': 'Cotton is a fiber crop used in textiles. It requires hot, dry climates and fertile soil.',
        'season': 'Kharif',
        'harvest_time': '150-180 days',
        'emoji': '🌱'
    },
    'grapes': {
        'description': 'Grapes are used for wine and eating. They need sunny locations with good air circulation.',
        'season': 'Summer',
        'harvest_time': '2-3 years',
        'emoji': '🍇'
    },
    'jute': {
        'description': 'Jute is a fiber crop for textiles. It grows in warm, humid conditions with heavy rainfall.',
        'season': 'Kharif',
        'harvest_time': '120-150 days',
        'emoji': '🌾'
    },
    'kidneybeans': {
        'description': 'Kidney beans are nutritious legumes. They thrive in warm weather with adequate moisture.',
        'season': 'Kharif',
        'harvest_time': '90-120 days',
        'emoji': '🫘'
    },
    'lentil': {
        'description': 'Lentils are protein-rich pulses. They prefer cool, dry climates and well-drained soil.',
        'season': 'Rabi',
        'harvest_time': '80-110 days',
        'emoji': '🫘'
    },
    'maize': {
        'description': 'Maize (corn) is a staple grain crop. It needs warm temperatures and plenty of sunlight.',
        'season': 'Kharif',
        'harvest_time': '90-120 days',
        'emoji': '🌽'
    },
    'mango': {
        'description': 'Mangoes are sweet tropical fruits. They require hot, humid climates and well-drained soil.',
        'season': 'Summer',
        'harvest_time': '3-5 years',
        'emoji': '🥭'
    },
    'mothbeans': {
        'description': 'Mothbeans are drought-resistant legumes. They grow in arid regions with minimal water.',
        'season': 'Kharif',
        'harvest_time': '75-90 days',
        'emoji': '🫘'
    },
    'mungbean': {
        'description': 'Mungbeans are fast-growing legumes. They prefer warm, humid conditions.',
        'season': 'Kharif',
        'harvest_time': '60-75 days',
        'emoji': '🫘'
    },
    'muskmelon': {
        'description': 'Muskmelons are sweet melons. They need warm, sunny conditions and fertile soil.',
        'season': 'Summer',
        'harvest_time': '70-90 days',
        'emoji': '🍈'
    },
    'orange': {
        'description': 'Oranges are citrus fruits rich in vitamin C. They thrive in subtropical climates.',
        'season': 'Winter',
        'harvest_time': '3-4 years',
        'emoji': '🍊'
    },
    'papaya': {
        'description': 'Papayas are tropical fruits. They grow quickly in warm, humid environments.',
        'season': 'Year-round',
        'harvest_time': '6-12 months',
        'emoji': '🍈'
    },
    'pigeonpeas': {
        'description': 'Pigeonpeas are drought-tolerant legumes. They suit semi-arid regions.',
        'season': 'Kharif',
        'harvest_time': '150-180 days',
        'emoji': '🫘'
    },
    'pomegranate': {
        'description': 'Pomegranates are antioxidant-rich fruits. They prefer hot, dry climates.',
        'season': 'Fall',
        'harvest_time': '2-3 years',
        'emoji': '🍎'
    },
    'rice': {
        'description': 'Rice is a staple cereal crop. It requires flooded fields and warm, humid conditions.',
        'season': 'Kharif',
        'harvest_time': '120-150 days',
        'emoji': '🌾'
    },
    'watermelon': {
        'description': 'Watermelons are refreshing summer fruits. They need hot, sunny weather and sandy soil.',
        'season': 'Summer',
        'harvest_time': '70-90 days',
        'emoji': '🍉'
    }
}

# Load the trained model
try:
    model = joblib.load('models/RandomForest.pkl')
    logging.info("Model loaded successfully")
except Exception as e:
    model = None
    logging.error(f"Error loading model: {e}")

@app.route('/')
def home():
    """Renders the main input form page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request and returns the result page."""
    if model is None:
        return "Error: Model not loaded. Please check the model file.", 500

    try:
        # Get data from the form
        data = request.form.to_dict()
        input_data = np.array([[
            float(data['Nitrogen']),
            float(data['Phosphorus']),
            float(data['Potassium']),
            float(data['Temperature']),
            float(data['Humidity']),
            float(data['Soil_pH']),
            float(data['Rainfall'])
        ]])
        
        # Make predictions
        prediction_probs = model.predict_proba(input_data)[0]
        confidence = np.max(prediction_probs) * 100
        predicted_crop_index = np.argmax(prediction_probs)
        predicted_crop = crop_classes[predicted_crop_index]
        
        # Get crop information
        crop_data = crop_info.get(predicted_crop, {
            'description': 'Description not available.',
            'season': 'N/A',
            'harvest_time': 'N/A',
            'emoji': '🌱'
        })

        # Get top 3 alternative crops
        top_indices = np.argsort(prediction_probs)[-4:-1][::-1]
        alternatives = [
            {
                'name': crop_classes[i].title(),
                'confidence': f"{prediction_probs[i] * 100:.1f}%",
                'emoji': crop_info.get(crop_classes[i], {}).get('emoji', '🌱')
            }
            for i in top_indices
        ]

        # Get feature importances
        feature_importances = model.feature_importances_
        key_factors = [
            {
                'name': feature_names[i].upper(),
                'value': round(feature_importances[i], 3),
                'input_value': input_data[0][i]
            }
            for i in range(len(feature_names))
        ]
        key_factors = sorted(key_factors, key=lambda x: x['value'], reverse=True)[:5]
        max_importance = key_factors[0]['value'] if key_factors else 1.0

        return render_template(
            'result.html',
            crop_name=predicted_crop.title(),
            crop_emoji=crop_data['emoji'],
            confidence=f"{confidence:.1f}%",
            crop_description=crop_data['description'],
            season=crop_data['season'],
            harvest_time=crop_data['harvest_time'],
            key_factors=key_factors,
            max_importance=max_importance,
            alternatives=alternatives,
            input_data={
                'nitrogen': data['Nitrogen'],
                'phosphorus': data['Phosphorus'],
                'potassium': data['Potassium'],
                'temperature': data['Temperature'],
                'humidity': data['Humidity'],
                'ph': data['Soil_pH'],
                'rainfall': data['Rainfall']
            }
        )

    except (ValueError, KeyError) as e:
        logging.error(f"Input error: {e}")
        return f"Invalid input. Please ensure all fields are filled with numeric values. Error: {e}", 400
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return f"An unexpected error occurred: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)