import os
import glob
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

#------------------------------------------------------------------------------------------------------------#

#-----------------------------------تحميل ملفات الصوت واستخراج الميزات---------------------------------------#
# Function to load audio files
def load_audio_files(base_path):
    features = []
    labels = []
    accents = ['Hebron', 'Jerusalem', 'Nablus','Ramallah-Reef']

    for accent in accents:
        folder_path = os.path.join(base_path, accent)
        for file_path in glob.glob(os.path.join(folder_path, '*.wav')):
            y, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfccs_mean = np.mean(mfccs, axis=1)
            features.append(mfccs_mean)
            labels.append(accent)

    return np.array(features), np.array(labels)

#-----------------------------------------------------------------------------------------------------------#

#--------------------------------------------تقسيم البيانات-------------------------------------------------#
train_base_path = '/Users/mahmoudatia/Desktop/Spoken/Train'
test_base_path = '/Users/mahmoudatia/Desktop/Spoken/Test'

# Load and preprocess the training dataset
X_train, y_train = load_audio_files(train_base_path)
X_test, y_test = load_audio_files(test_base_path)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#----------------------------------------------------------------------------------------------------------#

#------------------------------------------تدريب نموذج SVM-------------------------------------------------#
# Use Grid Search to find the best parameters for SVM
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'class_weight': ['balanced']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model from Grid Search
best_svm = grid_search.best_estimator_

# Evaluate the model
y_pred = best_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred, labels=best_svm.classes_)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=best_svm.classes_, yticklabels=best_svm.classes_,cmap='binary', linewidths=1)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test, y_pred, target_names=best_svm.classes_))

#----------------------------------------------------------------------------------------------------------#

'''
#this is first test before using grid search on svm
#------------------------------------------تدريب نموذج SVM-------------------------------------------------#
# Train the SVM classifier
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Evaluate the model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# y_test: التسميات الحقيقية لمجموعة الاختبار.
# y_pred: التسميات المتوقعة بواسطة النموذج.

print(f'Accuracy: {accuracy * 100:.2f}%')

cm = confusion_matrix(y_test, y_pred, labels=svm.classes_)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=svm.classes_, yticklabels=svm.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test, y_pred, target_names=svm.classes_))
#----------------------------------------------------------------------------------------------------------#
'''

#----------------------------------------حفظ النموذج والمقياس----------------------------------------------#
# Save the trained model and scaler for future use
with open('best_svm_model.pkl', 'wb') as model_file:
    pickle.dump(best_svm, model_file)
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Function to predict the accent of a new audio file
def predict_accent(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1).reshape(1, -1)
    mfccs_scaled = scaler.transform(mfccs_mean)
    predicted_accent = best_svm.predict(mfccs_scaled)
    return predicted_accent[0]

# Example
test_audio_path = '/Users/mahmoudatia/Desktop/Desktop/hebron1.wav'
predicted_accent = predict_accent(test_audio_path)
print(f'The predicted accent for the test audio is: {predicted_accent}')


#-----------------------------------------------------SVM--------------------------------------------------------#
#                       accuracy (70%) (n_MFCCs = 13)                                                            #
#                                   |                                                                            #
#                                   |   development n_MFCCs , accueacy does no change but predicted changed      #
#                                   |                                                                            #
#                       accuracy (70%) (n_MFCCs = 20)                                                            #
#                                   |                                                                            #
#                                   |   development SVM by using grid search                                     #
#                                   |                                                                            #
#                       accuracy (75%) (n_MFCCs = 13)                                                            #
#                                   |                                                                            #
#                                   |   development n_MFCCs , accueacy changed and predicted changed             #
#                                   |                                                                            #
#                       accuracy (80%) (n_MFCCs = 20)                                                            #
#                                   |                                                                            #
#                                   |   development SVM by using grid search and change n_MFCCs                  #
#                                   |                                                                            #
#                            accuracy = 80%                                                                      #
#----------------------------------------------------------------------------------------------------------------#
