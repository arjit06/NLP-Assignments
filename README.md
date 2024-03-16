# NLP-Assignments

## Assignment 1: Language Modelling
The assignment consists of 2 tasks:-
1. Implementation of tokenization using BytePair encoding.
2. Creation of a Bigram Language Model (LM), and modification of the standard implementation of Bigram LM to generate emotion-oriented sentences

The link to the dataset is https://drive.google.com/drive/folders/1x0CYAM8-kfA8j2m3ftWNsHfS3XXsdrXW. It is a subset of the Twitter Emotion Dataset from Hugging Face.

In the first task Byte Pair Encoding was implemented from Scratch. In the second task, a Bigram Language Model was created. Then two smoothing algorithms were implemented: Laplace and Knesner-Ney. Then using the emotion scores (of the transformers library), the standard probability of the Bigram model was modified to create an Emotion-sensitive Bigram Model. 50 samples corresponding to each of the 6 emotions (joy, surprise, anger, love, fear, sadness) were created. Then extrinsic evaluation was carried out with the original corpus as the training data, and the generated samples as the testing data (labels for the generated samples will be the emotion corresponding to which we generated each sample). An SVC model was trained from the Scikit-Learn library, using the TF-IDF vectorizer for vectorizing the text samples. Grid Search was then conducted to find out the best parameters. The best accuracy was found to be 70%. 
<br><br><br>

## Assignment 2: Named Entity Recognition and Aspect Term Extraction

### Task 1
This task corresponds to Legal Named Entities Recognition from the Indian court judgment text. Named entities refer to the key subjects (given as labels in the dataset) of a piece of text. This is the link to the dataset: https://drive.google.com/drive/folders/1qG_fkEMx69V10W_K8xLh80ZVaQv6Pt26 . The task is implemented using the methods described below as Parts. 
![image](https://github.com/arjit06/NLP-Assignments/assets/108218688/d92275a0-566c-4ebe-bc5a-a40392405923)
<br><br>

### Task 2 
This task corresponds to Aspect Term Extraction from Laptop-review texts. Aspect extraction is the task of identifying and extracting terms relevant to opinion mining and sentiment analysis (e.g., terms for laptop features.This is the link to the dataset: https://drive.google.com/drive/folders/1QHKT0UbsnosACLNxdicbeHoXtDS1l56f . This task is also implemented using the methods described below as Parts. 
![image](https://github.com/arjit06/NLP-Assignments/assets/108218688/9cf85945-56cf-46ff-bdbd-44bb7c526ff3)
<br><br>

### Part 1: Data Preparation
The JSON files of both the datasets are preprocessed and appropriate Training, Validation, and Testing splits are created wherever required. Tokenization is done based on space and BIO chunking is done for both datasets. Task 1 has 13 classes (namely- COURT, PETITIONER, RESPONDENT, JUDGE, LAWYER, DATE, ORG, GPE, STATUTE, PRECEDENT, CASE NUMBER, WITNESS, OTHER_PERSON) so in total 27 labels (acc to BIO encoding). Likewise, Task 2 just has 3 classes (B, I ,O) for the aspect term. 
<br><br>

### Part 2: Baseline models
Implementation of RNN-based sequence tagging models using the above datasets  in the following setups:

<ol>
<li> Model 1: Using vanilla RNN layer </li>
<li> Model 2: Using LSTM network </li>
<li> Model 3: Using GRU network </li>
</ol>

In each of these setups, three different pre-trained word embeddings were used: word2vec, GloVe, and fasttext in the embedding layer of the networks and total of nine models were trained for each dataset.
<br><br>

### Part 3: BiLSTM-CRF Model
A BiLSTM-CRF model (Model 4) was implemented for token classification using the above dataset (formed in Part 1A). The same three embeddings were used 


The following plots for every model were generated: 
<ul>
<li>Loss Plot: Training Loss and Validation Loss V/s Epochs</li>
<li>F1 Plot: Training Macro-F1-score and Validation Macro-F1-score V/s Epochs</li>
<li>Analysis and Explanation of the plots obtained</li>
</ul>
<br><br>


The following were the results of the implementations:- 
<br><br>
<b><u>Dataset 1: Named Entity Recognition</u></b>

<table>
  <tr>
    <th>Model</th>
    <th>Embedding</th>
    <th>Macro F1</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td>Vanilla RNN</td>
    <td>Word2Vec</td>
    <td>0.61</td>
    <td>93.7%</td>
  </tr>
  <tr>
    <td>LSTM</td>
    <td>Word2Vec</td>
    <td>0.62</td>
    <td>94%</td>
  </tr>
  <tr>
    <td>GRU</td>
    <td>Word2Vec</td>
    <td>0.64</td>
    <td>94%</td>
  </tr>
  <tr>
    <td>Bi-LSTM CRF</td>
    <td>Word2Vec</td>
    <td>0.74</td>
    <td>96%</td>
  </tr>
  <tr>
    <td>Vanilla RNN</td>
    <td>Glove</td>
    <td>0.64</td>
    <td>94%</td>
  </tr>
  <tr>
    <td>LSTM</td>
    <td>Glove</td>
    <td>0.56</td>
    <td>93%</td>
  </tr>
  <tr>
    <td>GRU</td>
    <td>Glove</td>
    <td>0.56</td>
    <td>93%</td>
  </tr>
  <tr>
    <td>Bi-LSTM CRF</td>
    <td>Glove</td>
    <td>0.66</td>
    <td>95%</td>
  </tr>
  <tr>
    <td>Vanilla RNN</td>
    <td>FastText</td>
    <td>0.68</td>
    <td>96%</td>
  </tr>
  <tr>
    <td>LSTM</td>
    <td>FastText</td>
    <td>0.69</td>
    <td>96%</td>
  </tr>
  <tr>
    <td>GRU</td>
    <td>FastText</td>
    <td>0.68</td>
    <td>95%</td>
  </tr>
  <tr>
    <td>Bi-LSTM CRF</td>
    <td>FastText</td>
    <td>0.77</td>
    <td>96%</td>
  </tr>
</table>


<br><br>
<b><u>Dataset 2: Aspect Term Identification</u></b>

<table>
  <tr>
    <th>Model</th>
    <th>Embedding</th>
    <th>Macro F1</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td>Vanilla RNN</td>
    <td>Word2Vec</td>
    <td>0.54</td>
    <td>77%</td>
  </tr>
  <tr>
    <td>LSTM</td>
    <td>Word2Vec</td>
    <td>0.52</td>
    <td>77%</td>
  </tr>
  <tr>
    <td>GRU</td>
    <td>Word2Vec</td>
    <td>0.5</td>
    <td>77%</td>
  </tr>
  <tr>
    <td>Bi-LSTM CRF</td>
    <td>Word2Vec</td>
    <td>0.76</td>
    <td>95%</td>
  </tr>
  <tr>
    <td>Vanilla RNN</td>
    <td>Glove</td>
    <td>0.55</td>
    <td>77%</td>
  </tr>
  <tr>
    <td>LSTM</td>
    <td>Glove</td>
    <td>0.54</td>
    <td>77%</td>
  </tr>
  <tr>
    <td>GRU</td>
    <td>Glove</td>
    <td>0.55</td>
    <td>77%</td>
  </tr>
  <tr>
    <td>Bi-LSTM CRF</td>
    <td>Glove</td>
    <td>0.77</td>
    <td>97%</td>
  </tr>
  <tr>
    <td>Vanilla RNN</td>
    <td>FastText</td>
    <td>0.54</td>
    <td>77%</td>
  </tr>
  <tr>
    <td>LSTM</td>
    <td>FastText</td>
    <td>0.55</td>
    <td>77%</td>
  </tr>
  <tr>
    <td>GRU</td>
    <td>FastText</td>
    <td>0.54</td>
    <td>77%</td>
  </tr>
  <tr>
    <td>Bi-LSTM CRF</td>
    <td>FastText</td>
    <td>0.74</td>
    <td>97%</td>
  </tr>
</table>
