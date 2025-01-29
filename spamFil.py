import string
import argparse
import os
import logging
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Naplózás beállítása
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Stopword-ok letöltése (szükséges az NLTK használatához)
nltk.download('stopwords')

# Változók inicializálása
stemmer = PorterStemmer()  # Szótövező (PorterStemmer)
stopwords_set = set(stopwords.words('english'))  # Angol stopword-ok halmaza
vectorizer = None  # A vektorizáló objektum, később betöltésre kerül
clf = None  # A modell helyőrzője
x_train, x_test, y_train, y_test = None, None, None, None  # A train-test adathalmazok helyőrzői

# Fájl elérési utak
preprocessed_data_file = 'preprocessed_data.joblib'  # Előfeldolgozott adatok mentése ide
model_file = 'spam_model.joblib'  # A modell mentése ide
vectorizer_file = 'vectorizer.joblib'  # A vektorizáló mentése ide


def preprocess_data():
    """Az adatok előfeldolgozása, vektorizáló betanítása és az előfeldolgozott adatok mentése."""
    logging.info("Adatok előfeldolgozása...")
    df = pd.read_csv('spam_ham_dataset.csv')  # Beolvassuk a CSV fájlt

    # Ellenőrizzük, hogy a CSV fájl tartalmazza-e a szükséges oszlopokat
    if 'text' not in df.columns or 'label_num' not in df.columns:
        raise ValueError("A bemeneti adatfájlnak tartalmaznia kell 'text' és 'label_num' oszlopokat.")

    # Az e-mailek szövegének tisztítása (új sor karakterek eltávolítása)
    df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ' '))

    # Szövegek előfeldolgozása
    corpus = []  # Itt tároljuk az előfeldolgozott szövegeket
    for i in range(len(df)):
        text = df['text'].iloc[i].lower()  # Kisbetűssé alakítás
        text = text.translate(str.maketrans('', '', string.punctuation)).split()  # Írásjelek eltávolítása
        text = [stemmer.stem(word) for word in text if word not in stopwords_set]  # Szótövezés és stopword-ok eltávolítása
        text = ' '.join(text)  # Szavak összefűzése szöveggé
        corpus.append(text)  # Az előfeldolgozott szöveg hozzáadása a listához

    # Vektorizáló betanítása
    global vectorizer
    vectorizer = CountVectorizer()  # Szövegek vektorokká alakítása
    x = vectorizer.fit_transform(corpus).toarray()  # Szövegek vektorizálása
    y = df.label_num  # Címkék kinyerése

    # Előfeldolgozott adatok és vektorizáló mentése joblib segítségével
    joblib.dump((x, y), preprocessed_data_file)
    logging.info(f"Előfeldolgozott adatok mentve ide: {preprocessed_data_file}")

    joblib.dump(vectorizer, vectorizer_file)
    logging.info(f"Vektorizáló mentve ide: {vectorizer_file}")

    logging.info("Adatok előfeldolgozva és mentve!")


def load_preprocessed_data():
    """Előfeldolgozott adatok és vektorizáló betöltése, vagy előfeldolgozás indítása, ha hiányzik."""
    global x_train, x_test, y_train, y_test, vectorizer
    if os.path.exists(preprocessed_data_file) and os.path.exists(vectorizer_file):
        logging.info("Előfeldolgozott adatok és vektorizáló betöltése...")
        x, y = joblib.load(preprocessed_data_file)  # Előfeldolgozott adatok betöltése
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # Train-test felosztás

        vectorizer = joblib.load(vectorizer_file)  # Vektorizáló betöltése
        logging.info("Előfeldolgozott adatok és vektorizáló betöltve!")
    else:
        logging.info("Előfeldolgozott adatok nem találhatók. Előfeldolgozás indítása...")
        preprocess_data()  # Előfeldolgozás indítása
        load_preprocessed_data()  # Adatok újratöltése


def train_model():
    """Modell betanítása és mentése."""
    global clf
    if x_train is None or y_train is None:
        logging.error("Kérlek, először töltsd be és előfeldolgozd az adatokat.")
        return

    logging.info("Modell betanítása...")
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)  # Random Forest modell létrehozása
    clf.fit(x_train, y_train)  # Modell betanítása

    # Modell mentése
    joblib.dump(clf, model_file)
    logging.info(f"Modell mentve ide: {model_file}")
    logging.info("Modell betanítva és mentve!")


def load_model():
    """Mentett modell és vektorizáló betöltése."""
    global clf, vectorizer
    if os.path.exists(model_file) and os.path.exists(vectorizer_file):
        logging.info("Modell és vektorizáló betöltése...")
        clf = joblib.load(model_file)  # Modell betöltése
        vectorizer = joblib.load(vectorizer_file)  # Vektorizáló betöltése
        logging.info("Modell és vektorizáló sikeresen betöltve!")
    else:
        logging.error(f"A modell vagy a vektorizáló nem található. Kérlek, győződj meg róla, hogy a következő fájlok léteznek: {model_file}, {vectorizer_file}")
        logging.error("Kérlek, először tanítsd be a modellt.")


def evaluate_model():
    """Modell kiértékelése."""
    if clf is None:
        logging.error("Kérlek, először tanítsd be a modellt.")
        return

    logging.info("Modell kiértékelése...")
    y_pred = clf.predict(x_test)  # Predikciók készítése a teszt adatokon
    accuracy = accuracy_score(y_test, y_pred)  # Pontosság számítása
    report = classification_report(y_test, y_pred)  # Osztályozási jelentés

    logging.info(f"Pontosság: {accuracy}")
    logging.info("Osztályozási jelentés:")
    logging.info(report)


def predict_email(email: str):
    """E-mail spam vagy ham osztályozása."""
    if clf is None or vectorizer is None:
        logging.error("Kérlek, először tanítsd be a modellt.")
        return

    # E-mail szövegének előfeldolgozása
    email = email.lower().translate(str.maketrans('', '', string.punctuation)).split()  # Kisbetűssé alakítás és írásjelek eltávolítása
    email = [stemmer.stem(word) for word in email if word not in stopwords_set]  # Szótövezés és stopword-ok eltávolítása
    email = ' '.join(email)  # Szavak összefűzése

    # E-mail vektorizálása
    email_vector = vectorizer.transform([email]).toarray()  # Szöveg vektorrá alakítása
    prediction = clf.predict(email_vector)  # Predikció készítése
    label = "Spam" if prediction[0] == 1 else "Ham"  # Eredmény értelmezése

    logging.info(f"Az e-mail osztályozása: {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spam detektáló program")
    parser.add_argument("--train", action="store_true", help="Modell betanítása")
    parser.add_argument("--evaluate", action="store_true", help="Modell kiértékelése")
    parser.add_argument("--predict", type=str, help="E-mail spam vagy ham osztályozása")
    args = parser.parse_args()

    # Adatok betöltése és előfeldolgozása minden művelet előtt
    load_preprocessed_data()

    # Parancsok kezelése
    if args.train:
        train_model()
    elif args.evaluate:
        load_model()  # Modell betöltése kiértékelés előtt
        evaluate_model()
    elif args.predict:
        load_model()  # Modell betöltése predikció előtt
        predict_email(args.predict)
    else:
        logging.info("Nincs művelet megadva. Használd a --help kapcsolót a használati útmutatóhoz.")