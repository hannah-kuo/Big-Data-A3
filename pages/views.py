import json
import pickle
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Load the logistic regression model and scaler
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def index(request):
    return render(request, 'helloworld/index.html')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        feature_values = [
            float(data['long_hair']),
            float(data['forehead_width_cm']),
            float(data['forehead_height_cm']),
            float(data['nose_wide']),
            float(data['nose_long']),
        ]

        # Scale the features and make the prediction
        scaled_features = scaler.transform([feature_values])
        prediction = model.predict(scaled_features)

        print("Prediction result:", prediction[0])  # Print the prediction result to the terminal

        # Return the prediction result as a JSON response
        return JsonResponse({'prediction': prediction[0]})
    else:
        return JsonResponse({"error": "Invalid request method"})
