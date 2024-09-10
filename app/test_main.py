from fastapi.testclient import TestClient
from app.main import app

# Initialize the TestClient with your FastAPI app
client = TestClient(app)

def test_hateful_predict():
    # Define the data to send to the POST request
    test_data = {
        "text": "Salut fils de pute !!!"
    }

    # Make a POST request to the /predict/ endpoint
    response = client.post("/predict/", json=test_data)

    # Check that the response status code is 200
    assert response.status_code == 200

    # Check that the response contains the "predicted_class" key
    json_data = response.json()
    assert "predicted_class" in json_data
    assert isinstance(json_data["predicted_class"], str) 
    assert json_data["predicted_class"] == "hateful" # Assert it is hateful

def test_non_hateful_predict():
    # Define the data to send to the POST request
    test_data = {
        "text": "Salut, comment vas-tu ?"
    }

    # Make a POST request to the /predict/ endpoint
    response = client.post("/predict/", json=test_data)

    # Check that the response status code is 200
    assert response.status_code == 200

    # Check that the response contains the "predicted_class" key
    json_data = response.json()
    assert "predicted_class" in json_data
    assert isinstance(json_data["predicted_class"], str)
    assert json_data["predicted_class"] == "non-hateful" # Assert it is not hateful
    
def test_predict_invalid_data():
    # Send an empty string
    test_data = {"text": ""}
    response = client.post("/predict/", json=test_data)
    assert response.status_code == 200  # Still processes, but check predicted_class

    # Send invalid JSON
    response = client.post("/predict/", json="Invalid data")
    assert response.status_code == 422  # Unprocessable Entity for invalid input