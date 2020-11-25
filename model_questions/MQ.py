import pickle
import os

class MQ():
        def __init__(self, clf, vect):
            self.clf = clf
            self.vect = vect
            self.labels = {0: 'Что вы под этим имеете в виду?',
                        1: 'А для чего вам это?',
                        2: 'Можете ли вы выделить какие-то этапы для достижения указанной цели?',
                        3: 'Как эта цель связана с окружающими вас людьми? С событиями и явлениями вашей жизни?',
                        4: 'Что в этой цели для вас наиболее приоритетно?',
                        6: 'Как вы можете использовать для этого свой имеющийся опыт?',
                        7: 'Как вы поймете, что вы достигли этой цели? как будет выглядеть результат?',
                        8: 'Что бы вы хотели делать с этой целью дальше?',
                        9: 'Что могло бы стать первым шагом для достижения данной цели?',
                        10: 'Какой смысл вы в это вкладываете?'}

        def predict(self, text):
            vectorized = self.vect.transform([text])
            predictions = self.clf.predict(vectorized)
            return set(self.convert(predictions))

        def convert(self, x):
                out = []
                for i in x[0]:
                    out.append(self.labels[i])
                return out

def load_questions_model(path=""):
    with open(os.path.join(path, "model_questions.pkl"), "rb") as f:
        clf = pickle.load(f)

    with open(os.path.join(path, "model_questions_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    return MQ(clf, vectorizer)
