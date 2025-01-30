from flask import Flask, request, render_template
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure required NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))  # TF-IDF Vectorizer
    model = pickle.load(open('model.pkl', 'rb'))  # ML Model
except FileNotFoundError as e:
    raise Exception(f"Required file not found: {e.filename}. Ensure 'vectorizer.pkl' and 'model.pkl' exist.")

# Initialize the PorterStemmer
ps = PorterStemmer()


# Function to preprocess the text
def transform_text(text):
    """
    Transforms the input text by:
    - Lowercasing
    - Tokenizing
    - Removing stopwords and punctuation
    - Stemming the words
    """
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    text = [
        word for word in text if word not in stopwords.words('english') and word not in string.punctuation
    ]

    # Apply stemming to each word
    text = [ps.stem(word) for word in text]

    # Return processed text
    return " ".join(text)


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Ensure your index.html template exists!


# Route to handle spam prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get message input from the user
            input_sms = request.form['message']

            # Transform the text using the preprocessing function
            transformed_sms = transform_text(input_sms)

            # Convert transformed text into TF-IDF format
            vector_input = tfidf.transform([transformed_sms])

            # Predict using the loaded model
            result = model.predict(vector_input)[0]

            # Determine prediction result
            prediction = "Spam" if result == 1 else "Not Spam"
        except Exception as e:
            prediction = f"Error occurred: {e}"

        # Return and render prediction to web page
        return render_template('index.html', prediction=prediction)


# Run the application
if __name__ == "__main__":
    app.run(debug=True)
