import numpy as np
from collections import defaultdict
import utils

class BigramLMWithEmotion:
    def __init__(self, start_token='<s>', end_token='<eos>'):
        self.vocab = set()
        self.data = None
        self.start_token = start_token
        self.end_token = end_token
        self.bigram_counts = {}
        self.n_bigrams = 0
        self.unigram_counts = defaultdict(int)
        self.bigram_probs = {}
        self.bigram_emotion_probs = {}
        self.smoothing_method = None
        self.emotion_classifier = utils.emotion_scores


    
    def learn_vocabulary(self, data, smoothing_method=None, with_emotion=False):
        self.data = data
        self.smoothing_method = smoothing_method
        self._count_bigrams()
        self._estimate_probabilities()
        if with_emotion:
            self._modify_bigram_probs_with_emotions()



    def _count_bigrams(self):
        self.n_bigrams = 0
        #compute bigram counts
        for sentence in self.data:
            tokens = sentence.split()
            tokens = [self.start_token] + tokens + [self.end_token]

            self.vocab.update(tokens)

            for i in range(len(tokens) - 1):
                current_bigram = (tokens[i],tokens[i+1])
                self.unigram_counts[tokens[i]]+=1
                self.bigram_counts[current_bigram] = self.bigram_counts.get(current_bigram, 0) + 1
                self.n_bigrams += 1

            self.unigram_counts[tokens[len(tokens)-1]]+=1


    
    def _estimate_probabilities(self):
        if self.smoothing_method == 'laplace':
            self._laplace_smoothing()
        elif self.smoothing_method == 'kneser-ney':
            self._kneser_ney_smoothing()
        else:
            for bigram in self.bigram_counts:
                first_word = bigram[0]
                count=self.bigram_counts[bigram]
                probability=count/ self.unigram_counts[first_word]
                self.bigram_probs[bigram]=probability



    def _laplace_smoothing(self):
        V = len(self.vocab)
        for bigram in self.bigram_counts:
            first_word = bigram[0]
            curr_bigram_count=self.bigram_counts[bigram]
            smoothed_prob=(curr_bigram_count+1)/(self.unigram_counts[first_word]+V)
            self.bigram_probs[bigram]=smoothed_prob


    def _kneser_ney_smoothing(self):
        discount = 0.5
        n_bigrams = self.n_bigrams

        for bigram in self.bigram_counts:
            first_word,second_word = bigram
            cnt_bigram = self.bigram_counts[bigram]
            cnt_unigram = self.unigram_counts[first_word]

            cont_cnt = len(set([w1 for (w1,w2) in self.bigram_counts.keys() if w2==second_word]))
            alpha_cnt = len(set([w2 for (w1,w2) in self.bigram_counts.keys() if w1==first_word]))
            p_cont = cont_cnt/n_bigrams
            alpha = (discount/cnt_unigram)*(alpha_cnt)

            prob = (max(0,cnt_bigram - discount))/cnt_unigram + alpha*p_cont
            self.bigram_probs[bigram] = prob



    def generate_text(self,emotion,max_length=20):
        generated_text = [self.start_token]

        for _ in range(max_length):
            next_word = self.generate_next_word(generated_text[-1],emotion)
            if next_word == self.end_token:
                break
            generated_text.append(next_word)

        return ' '.join(generated_text)


    def generate_next_word(self, prev_word, emotion):
        suggestions = [(next_word,self.bigram_emotion_probs[(prev_word,next_word)][emotion]) for (prev_word,next_word) in self.bigram_probs.keys()]
        
        if(len(suggestions)==0):
            return self.end_token
        
        next_words,probabilities = zip(*suggestions)
        total = sum(probabilities)
        probabilities = list(probabilities)
        for i in range(len(probabilities)):
            probabilities[i] /= total
        next_word = np.random.choice(next_words,p=probabilities)
        return next_word



    def _modify_bigram_probs_with_emotions(self):
        for bigram in self.bigram_probs:
            first_word, second_word = bigram
            bigram_emotion_score = self.emotion_classifier(first_word+" "+second_word)
            unigram_emotion_score = self.emotion_classifier(first_word)

            total = 0.0

            self.bigram_emotion_probs[bigram] = defaultdict(float)
            for i,item in enumerate(bigram_emotion_score):
                self.bigram_emotion_probs[bigram][item['label']] = self.bigram_probs[bigram] + item['score']/unigram_emotion_score[i]['score']
                total += self.bigram_emotion_probs[bigram][item['label']]
            for key in self.bigram_emotion_probs[bigram].keys():
                self.bigram_emotion_probs[bigram][key] /= total



class SentenceGenerator:
    def __init__(self,emotions,bigram_model):
        self.emotions = emotions
        self.bigram_model = bigram_model
    
    def generate(self,n_sentences,min_len):
        out_text = {}
        for emotion in self.emotions:
            out_text[emotion] = []
            while(len(out_text[emotion])<n_sentences):
                sentence = self.bigram_model.generate_text(emotion)[4:]
                if(len(sentence.split(' '))<min_len):
                    continue
                out_text[emotion].append(sentence)
                # print(sentence)
        return out_text



emotions = ['sadness','joy','fear','love','anger','surprise']
n_sentences = 50
min_len = 7

generated_samples = SentenceGenerator(emotions,bigram_model).generate(n_sentences,min_len)
test_data = []
test_labels = []

!rm -rf generated_samples
!mkdir generated_samples

for emotion in emotions:
    path = "./generated_samples/gen_"+emotion+".txt"
    file = open(path,"w+")
    for sent in generated_samples[emotion]:
        file.write(sent+"\n")
        test_data.append(sent)
        test_labels.append(emotion)
    file.close()
    print(f"{emotion}: Generated!")



