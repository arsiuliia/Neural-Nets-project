from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow.compat.v1 as tf
import numpy as np
import pickle
import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = FastAPI()

# Accepter toutes les demandes du formulaire html
origins = ["*"]
methods = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=["*"],
)

#Définitions des clases pour la génération et la détection

#Une classe pour les news à détecter
class News(BaseModel):
    title: str

#Une classe pour le texte à générer
class TextGenerationRequest(BaseModel):
    seed: str

#####################################Partie détection############################################

#On appelle le modèle, les tokenizers et tout ce dont on a besoin pour la détection
#Au démarrage de l'api
@app.on_event("startup")
def charger_modele():
    global modele_detection, tokenizer1, max_length, padding_type, trunc_type
    modele_detection = load_model('modele_detection.h5')
    tokenizer1 = Tokenizer()
    tokenizer1.fit_on_texts([""])
    max_length = 54
    padding_type = 'post'
    trunc_type = 'post'
    

@app.post("/predict")
def predict(news: News):
    global modele_detection, tokenizer1, max_length, padding_type, trunc_type
    
    # Prétraitement des données
    sequences = tokenizer1.texts_to_sequences([news.title])
    sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Prédire si la nouvelle est fiable ou non
    prediction = modele_detection.predict(sequences)[0][0]
    
    # Retourner le résultat de la prédiction
    if prediction >= 0.5:
        return {"prediction": "This news is true"}
    else:
        return {"prediction": "This news is false"}
    
#################################Partie génération###############################################

#Charger le modèle et les poids

# Charger la conversion de caractères en entiers
with open('conv_gen_char2int.pickle', 'rb') as f:
    char2int = pickle.load(f)

# Charger la conversion d'entiers en caractères
with open('conv_gen_int2char.pickle', 'rb') as f:
    int2char = pickle.load(f)

# Charger le modèle de génération
modele_generation = load_model('modele_generation.h5')

#Fonction de génération utilisant le modèle
def generation(modele_generation,seed,char2int,int2char):
    s = seed
    sequence_length = 100       
    n_chars = 200
    vocab_size = len(char2int)
    generated = ""
    for i in tqdm.tqdm(range(n_chars), "Génération du texte"):
        X = np.zeros((1, sequence_length, vocab_size))
        for t, char in enumerate(seed):
            X[0, (sequence_length - len(seed)) + t, char2int[char]] = 1
        # Prédiction du caractère suivant
        predicted = modele_generation.predict(X, verbose=0)[0]
        # Conversion du vecteur en entier
        next_index = np.argmax(predicted)
        # Conversion de l'entier en caractère
        next_char = int2char[next_index]
        # Ajout du caractère à la génération
        generated += next_char
        seed = seed[1:] + next_char
    return(f"Seed: {s} \nGenerated text:\n{generated}")

@app.post("/text_generation")
def generation_texte(request: TextGenerationRequest):
    s = request.seed.lower()
    result = generation(modele_generation, s, char2int, int2char)
    return {"result": result}
