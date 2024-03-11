def print_top_5_bigrams(bigram_model):
    top_5_bigrams = sorted(bigram_model.bigram_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("Top 5 Bigrams and Probabilities:")
    print("\n{:<20} {:<20}".format("Bigram", "Probability"))
    print("="*40)
    for bigram, probability in top_5_bigrams:
        print("{:<20} {:.4f}".format(str(bigram), probability))



bigram_nosmooth = BigramLMWithEmotion()
bigram_nosmooth.learn_vocabulary(data=corpus)
print_top_5_bigrams(bigram_nosmooth)


bigram_laplace = BigramLMWithEmotion()
bigram_laplace.learn_vocabulary(data=corpus,smoothing_method='laplace')
print_top_5_bigrams(bigram_laplace)


bigram_kneserney = BigramLMWithEmotion()
bigram_kneserney.learn_vocabulary(data=corpus,smoothing_method='kneser-ney')
print_top_5_bigrams(bigram_kneserney)


import pickle

with open('bigram_model.pkl','wb') as file:
    pickle.dump(bigram_model,file)
with open('bigram_model.pkl','rb') as file:
    loaded_model = pickle.load(file)


train_labels = []
for sent in corpus:
    scores = utils.emotion_scores(sent)
    mx = 0.0
    emot = ""
    for item in scores:
        if item['score']>mx:
            emot = item['label']
            mx = item['score']
    train_labels.append(emot)


from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
X_test = vectorizer.transform(test_data)

svc = SVC()
param_grid = {
    'kernel':['rbf','linear','sigmoid'],
    'C':[0.9,0.95,0.8,0.85],
    'class_weight':['balanced',None],
    'max_iter':[-1,400,500]
}

grid_search = GridSearchCV(estimator=svc,param_grid=param_grid,cv=5,scoring='accuracy')
grid_search.fit(X,train_labels)

best_params = (grid_search.best_params_)
print(best_params)

svc_best = grid_search.best_estimator_
y_pred = svc_best.predict(X_test)

print(classification_report(test_labels,y_pred))