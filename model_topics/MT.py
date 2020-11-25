import pickle
import os

class TopicModel():
    def __init__(self, clf, vect, labels):
        self.clf = clf
        self.vect = vect
        self.labels = labels

    def predict(self, text):
        vectorized = self.vect.transform([text])
        predictions = self.clf.predict(vectorized)[0]
        return predictions, self.labels[predictions]

def load_topics_model(path=""):
    with open(os.path.join(path, "model_topics.pkl"), "rb") as f:
            clf = pickle.load(f)

    with open(os.path.join(path, "model_topics_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)

    with open(os.path.join(path, "custom_library.pkl"), "rb") as f:
            labels = pickle.load(f)

    return TopicModel(clf, vectorizer, labels)