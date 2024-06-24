#---------------------------------------------- Library -----------------------------------------------------#
import os
import glob
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from concurrent.futures import ThreadPoolExecutor

#------------------------------------------------------------------------------------------------------------#

#------------------------------------------ استخراج الميزات -------------------------------------------------#

# Function to extract features from a single audio file
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=80)
        mfccs_mean = np.mean(mfccs, axis=1)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)

        # Combine all features
        combined_features = np.concatenate((mfccs_mean, chroma_mean))
        return combined_features
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

#------------------------------------------------------------------------------------------------------------#

#------------------------------------------ تحميل ملفات الصوت------------------------------------------------#

# Function to load audio files and extract acoustic features
def load_audio_files(base_path):
    features = []
    labels = []
    accents = ['Hebron', 'Jerusalem', 'Nablus', 'Ramallah_Reef']

    for accent in accents:
        folder_path = os.path.join(base_path, accent)
        files = glob.glob(os.path.join(folder_path, '*.wav'))

        with ThreadPoolExecutor() as executor:
            results = executor.map(extract_features, files)

        for file_path, feature in zip(files, results):
            if feature is not None:
                features.append(feature)
                labels.append(accent)

    return np.array(features), np.array(labels)

#-----------------------------------------------------------------------------------------------------------#

#--------------------------------------------تقسيم البيانات-------------------------------------------------#

# paths to training and testing data
train_base_path = '/Users/mahmoudatia/Desktop/Spoken/Train'
test_base_path = '/Users/mahmoudatia/Desktop/Spoken/Test'

# load and preprocess the datasets
X_train, y_train = load_audio_files(train_base_path)
X_test, y_test = load_audio_files(test_base_path)

# check if data is loaded correctly
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#----------------------------------------------------------------------------------------------------------#

#------------------------------------------تدريب النماذج --------------------------------------------------#

# define the models
models = {
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier()
}

# function to train and evaluate a model
def train_and_evaluate(model_name):
    model = models[model_name]

    # define parameter grids for each model
    param_grids = {
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'class_weight': ['balanced']},
        'Random Forest': {'n_estimators': [50, 100, 200], 'max_features': ['sqrt', 'log2', None]},
        'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    }

    # use Grid Search to find the best parameters for the selected model
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5)
    grid_search.fit(X_train, y_train)

    # get the best model from Grid Search
    best_model = grid_search.best_estimator_

    # evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # print confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=best_model.classes_, yticklabels=best_model.classes_, cmap='binary', linewidths=1)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

    print(classification_report(y_test, y_pred, target_names=best_model.classes_))

    # save the trained model and scaler for future use
    with open(f'best_{model_name.lower().replace(" ", "_")}_model.pkl', 'wb') as model_file:
        pickle.dump(best_model, model_file)

    return best_model

#----------------------------------------------------------------------------------------------------------#

# user input to select the model
selected_model_name = input("Select a model (SVM, Random Forest, KNN): ")
best_model = train_and_evaluate(selected_model_name)

#----------------------------------------------------------------------------------------------------------#

#----------------------------------------حفظ النموذج والمقياس----------------------------------------------#

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# function to predict the accent of a new audio file
def predict_accent(audio_path, model_name):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=80)
        mfccs_mean = np.mean(mfccs, axis=1).reshape(1, -1)
        # Extract Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1).reshape(1, -1)
        # Extract Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1).reshape(1, -1)
        # Combine all features
        combined_features = np.concatenate((mfccs_mean, chroma_mean, contrast_mean), axis=1)
        mfccs_scaled = scaler.transform(combined_features)
        
        # Load the appropriate model
        with open(f'best_{model_name.lower().replace(" ", "_")}_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)

        predicted_accent = model.predict(mfccs_scaled)
        return predicted_accent[0]
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Example
# test_audio_path = '/Users/mahmoudatia/Desktop/Desktop/hebron1.wav'
# predicted_accent = predict_accent(test_audio_path, selected_model_name)
# print(f'The predicted accent for the test audio is: {predicted_accent}')


#----------------------------------------------------------------------------------------------------------------#
#                       SVM accuracy (75%) (when use all feature)                                                #
#                                   | (only mfccc 70%)                                                           #
#                                   | (only Chroma 35%)                                                          #
#                                   || (mfcc & Chroma 70%)                                                       #
#                                   | (only spectral 40%)                                                        #
#                                                                                                                #
#                       RF accuracy (60%) (when use all feature)                                                 #
#                                   | (only mfccc 60%)                                                           #
#                                   | (only Chroma 35%)                                                          #
#                                   || (mfcc & Chroma 65%)                                                       #   
#                                   | (only spectral 30%)                                                        #
#                                                                                                                #
#                       KNN accuracy (60%) (when use all feature)                                                #          
#                                   | (only mfccc 60%)                                                           #              
#                                   | (only Chroma 40%)                                                          #
#                                   || (mfcc & Chroma 65%)                                                       #
#                                   | (only spectral 30%)                                                        #
#----------------------------------------------------------------------------------------------------------------#