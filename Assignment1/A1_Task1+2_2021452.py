import re
import os
from collections import defaultdict
import utils
import numpy as np

class Tokenizer:
  def __init__(self):
    self.vocab = defaultdict(int)
    self.all_tokens= defaultdict(int)  # all tokens from first to last
    self.merge_rules=[]
    self.final_tokens=[]         # only the final tokens



  def merge_vocabulary(self,pair, vocab_old):
    vocab_new = {}
    self.merge_rules.append(pair)
    bigram = re.escape(' '.join(pair))
    regex=re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

   # merge the old vocab based on the new merging rule
    for word in vocab_old:
        w_out = regex.sub(''.join(pair), word) #find all words where bigram was found in the old vocab
        all_tokens=w_out.split()
        for a in all_tokens: self.all_tokens[a]+=1
        vocab_new[w_out] = vocab_old[word]
    return vocab_new



  def learn_vocabulary(self, corpus, num_merges):
      ### create a list of strings
      data = corpus.split('\n')

      ### updating dict vocab with words and frequencies
      for line in data:
          for word in line.split():
            new_word=' '.join(list(word)) + ' $'
            self.vocab[new_word] += 1

            all_tokens=new_word.split()
            for a in all_tokens: self.all_tokens[a]+=1


      ### making pairs and updating their frequencies
      for _ in range(num_merges):
          pairs = defaultdict(int)
          for word,freq in self.vocab.items():
              chars = word.split()
              for j in range(len(chars)-1):
                  pairs[chars[j],chars[j+1]] += freq


          best_pair = max(pairs, key=pairs.get)
          self.vocab = self.merge_vocabulary(best_pair, self.vocab)

      t = " ".join(list(self.vocab.keys()))
      d=defaultdict(int)
      l=t.split()
      for a in l: d[a]+=1
      self.final_tokens=list(d.keys())



  def tokenize(self,text_lists):
    # divide the text into individual letters

     ans_list=[]
     for a in text_lists:
          text=a
          data = text.split('\n')
          new_text=""
          for line in data:
              for word in line.split():
                  new_word=' '.join(list(word)) + ' $ '
                  new_text+=new_word

          # Tokenize the text based on merge rules
          merge_rules=self.merge_rules
          for rule in merge_rules:
              merged_token = "".join(rule)
              new_text = new_text.replace(" ".join(rule),merged_token)

          tokens = new_text.split()
          # print(tokens)
          # print()
          ans_list.append(tokens)


     return ans_list
  

  def write_to_file(self,root,text_list):
      token_path = os.path.join(root,"tokens.txt")
      rules_path = os.path.join(root,"merge_rules.txt")
      samples_path = os.path.join(root,"tokenized_samples.txt")

      file=open(token_path,"w+")
      for a in list(self.all_tokens.keys()):
          file.write(a+"\n")
      file.close()


      file=open(rules_path,"w+")
      for a in self.merge_rules:
          file.write(a[0]+","+a[1]+"\n")
      file.close()


      file=open(samples_path,"w")
      for a in text_list:
          s=",".join(a)+"\n"
          print(s)
          file.write(s)
      file.close()



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