from flask import Flask, request, render_template, redirect, url_for, session, flash
import os
import json
import shutil
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import unquote

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Paths to the dataset folders
IMAGE_FOLDER = 'dataset/floorplan_images'
TAG_FOLDER = 'dataset/human_annotated_tags'
SAVED_IMAGES_FOLDER = 'saved_floorplans'  # Folder to save images

# Ensure the 'saved_floorplans' directory exists
os.makedirs(SAVED_IMAGES_FOLDER, exist_ok=True)

# Main admin owner of the website
users = {'sraaz':'8','unknown':'1234','graha':'graha@123','model':'1234',}

# Load the dataset dynamically
def load_dataset(image_folder, tag_folder):
    data = []
    for tag_file in os.listdir(tag_folder):
        if tag_file.endswith('.txt') or tag_file.endswith('.json'):
            with open(os.path.join(tag_folder, tag_file), 'r') as f:
                tags = json.load(f)
                image_name = tag_file.replace('.txt', '.png').replace('.json', '.png')
                image_path = f"/static/floorplan_images/{image_name}"  # Correct static folder path
                data.append({
                    'image_path': image_path,
                    'bedrooms': tags.get('bedrooms', 0),
                    'bathrooms': tags.get('bathrooms', 0),
                    'kitchen': tags.get('kitchen', 0),
                    'living_room': tags.get('living_room', 0),
                    'garage': tags.get('garage', 0),
                    'entry': tags.get('entry', 0),
                    'sqft': tags.get('sqft', 0),
                    'prompt': tags.get('prompt', '')
                })
    return data

dataset = load_dataset(IMAGE_FOLDER, TAG_FOLDER)

# Prepare numerical data for KNN
numerical_data = np.array([
    [d['bedrooms'], d['bathrooms'], d['kitchen'], d['living_room'], d['garage'], d['entry'], d['sqft']]
    for d in dataset
])

# Prepare textual data for TF-IDF
tfidf_vectorizer = TfidfVectorizer()
prompts = [d['prompt'] for d in dataset]
tfidf_matrix = tfidf_vectorizer.fit_transform(prompts)

# Initialize KNN model
knn = NearestNeighbors(n_neighbors=4, metric='euclidean')
knn.fit(numerical_data)

# Routes for login and signup
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return render_template('index.html')  # Display index page after login
        else:
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            flash('Username already exists', 'error')
            return redirect(url_for('signup'))
        users[username] = password
        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        # Extract user inputs
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        kitchen = int(request.form['kitchen'])
        living_room = int(request.form['living_room'])
        garage = int(request.form['garage'])
        entry = int(request.form['entry'])
        sqft = int(request.form['sqft'])
        user_prompt = request.form['prompt']

        # Prepare numerical input
        user_input = np.array([bedrooms, bathrooms, kitchen, living_room, garage, entry, sqft]).reshape(1, -1)

        # Find closest matches based on numerical input
        distances, indices = knn.kneighbors(user_input)

        # Match user prompt to dataset prompts using cosine similarity
        user_prompt_vector = tfidf_vectorizer.transform([user_prompt])
        prompt_similarities = cosine_similarity(user_prompt_vector, tfidf_matrix).flatten()
        best_prompt_indices = prompt_similarities.argsort()[-4:][::-1]

        # Combine numerical and textual matches
        combined_indices = list(set(indices[0]) | set(best_prompt_indices))

        # Retrieve recommended results
        results = [{'image_path': dataset[i]['image_path'], 'prompt': dataset[i]['prompt']} for i in combined_indices]

        # Pass only 4 results to the template
        results = results[:4]

        return render_template('result.html', results=results, user_prompt=user_prompt)

@app.route('/save_image', methods=['POST'])
def save_image():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        # Get the image path from the form
        image_path = request.form['image_path']
        image_name = os.path.basename(unquote(image_path))  # Get image name from the URL

        # Create the path for saving the image
        save_path = os.path.join(SAVED_IMAGES_FOLDER, image_name)

        # Copy the image from static folder to the saved folder
        image_source = os.path.join('static', image_path.lstrip('/static/'))
        shutil.copy(image_source, save_path)

        return f"Image saved successfully as {image_name}"

if __name__ == '__main__':
    app.run(debug=True) 