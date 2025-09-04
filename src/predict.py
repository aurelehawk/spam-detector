import joblib
import os
from pathlib import Path

def predict_spams(text):
    """
    predit si le message est spam ou non

    Args: 
        text(str): le message a predire
    retrurns :
    str: "spam" ou "ham"
    """
    # load the trained pipeline (TF-IDF + LogisticRegression)
    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / "models" / "model.pkl"
    model = joblib.load(model_path)

    # prédire en utilisant le pipeline directement sur le texte brut           
    prediction = model.predict([text])[0]

    # retourner "spam" ou "ham"
    # le modèle a été entraîné avec des étiquettes texte ("spam"/"ham")
    return str(prediction)

# exemple d'application

if __name__ == "__main__":
    messages = [
        "Congratulations! You've won a $1000 gift card. Click here to claim your prize.", 
        "I'm sorry to inform you that your account has been suspended. Please contact support for more information.",
        "You have a new email from John Doe. Please check your inbox.",
        "Please click the link to reset your password."
    ] 
    
    print("Spam Detection Results :")
    print("-" * 50)

    # afficher les résultats de la prédiction pour chaque message
    for msg in messages:
        prediction = predict_spams(msg)
        print(f"Message: {msg[:50]}...")
        print(f"Prediction: {prediction}")
        print("-" * 50)

    
