import numpy as np
from collections import Counter
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from sentence_transformers import SentenceTransformer, util

class SeqEval:
  def __init__(self):
    self.results = {}
    self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    self.rouge = Rouge()

  def evaluate(self, candidates, references, verbose=True):

    self.compute_bleu(candidates, references)
    if verbose == True:
      print('***************')
      print('* BLEU SCORES *')
      print('***************')
      print('BLEU-1: ', self.results['bleu_1'])
      print('BLEU-2: ', self.results['bleu_2'])
      print('BLEU-3: ', self.results['bleu_3'])
      print('BLEU-4: ', self.results['bleu_4'])
      print('\n')

    self.compute_rouge(candidates, references)
    if verbose == True:
      print('****************')
      print('* ROUGE SCORES *')
      print('****************')
      print('ROUGE-1 PRECISION: ', self.results['rouge_1_precision'])
      print('ROUGE-1 RECALL: ', self.results['rouge_1_recall'])
      print('ROUGE-1 F1 : ', self.results['rouge_1_f1'])
      print('\n')
      print('ROUGE-2 PRECISION: ', self.results['rouge_2_precision'])
      print('ROUGE-2 RECALL: ', self.results['rouge_2_recall'])
      print('ROUGE-2 F1 : ', self.results['rouge_2_f1'])
      print('\n')
      print('ROUGE-L PRECISION: ', self.results['rouge_l_precision'])
      print('ROUGE-L RECALL: ', self.results['rouge_l_recall'])
      print('ROUGE-L F1 : ', self.results['rouge_l_f1'])
      print('\n')

    self.compute_distinct_n(candidates)
    if verbose == True:
      print('*********************')
      print('* DISTINCT-N SCORES *')
      print('*********************')
      print('INTER DIST-1: ', self.results['inter_dist1']) 
      print('INTER DIST-2: ', self.results['inter_dist2']) 
      print('INTRA DIST-1: ', self.results['intra_dist1']) 
      print('INTRA DIST-2: ', self.results['intra_dist2'])
      print('\n')

    self.compute_semantic_textual_similarity(candidates, references)
    if verbose == True:
      print('******************************************************')
      print('* SEMANTIC TEXTUAL SIMILARITY (Sentence Transformer) *')
      print('******************************************************')
      print('COSINE SIMILARITY: ', self.results['semantic_textual_similarity']) 

    return self.results

  def compute_bleu(self, candidates, references):
    """Counting matching n-grams in the candidate translation to n-grams in the reference text
    Args:
      candidates (list): list of generated outputs by the model
      references (list): list of ground-truth sentences
    Returns:
      bleu_1, bleu_2, bleu_3, bleu_4 scores
    """

    total_bleu_1 = 0
    total_bleu_2 = 0
    total_bleu_3 = 0
    total_bleu_4 = 0
    smoothie = SmoothingFunction().method4
    for i in range(len(candidates)):
      #Prepare inputs
      reference = references[i]
      reference = [reference.split()]
      candidate = candidates[i]
      candidate = candidate.split()
      #Compute BLEU
      bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)
      bleu_2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
      bleu_3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
      bleu_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
      #Add to total
      total_bleu_1 = total_bleu_1 + bleu_1
      total_bleu_2 = total_bleu_2 + bleu_2
      total_bleu_3 = total_bleu_3 + bleu_3
      total_bleu_4 = total_bleu_4 + bleu_4

    #Store results in dictionary
    self.results['bleu_1'] = total_bleu_1/(len(candidates))
    self.results['bleu_2'] = total_bleu_2/(len(candidates))
    self.results['bleu_3'] = total_bleu_3/(len(candidates))
    self.results['bleu_4'] = total_bleu_4/(len(candidates))

  def compute_rouge(self, candidates, responses):
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores
    Args:
      candidates (list): list of generated outputs by the model
      references (list): list of ground-truth sentences
    Returns:
      precision, recall, and f1-score for rouge-1, rouge-2, and rouge-l
    """

    rouge_scores = self.rouge.get_scores(candidates, responses, avg=True)
    self.results['rouge_1_precision'] = rouge_scores['rouge-1']['p']
    self.results['rouge_1_recall'] = rouge_scores['rouge-1']['r']
    self.results['rouge_1_f1'] = rouge_scores['rouge-1']['f']
    self.results['rouge_2_precision'] = rouge_scores['rouge-2']['p']
    self.results['rouge_2_recall'] = rouge_scores['rouge-2']['r']
    self.results['rouge_2_f1'] = rouge_scores['rouge-2']['f']
    self.results['rouge_l_precision'] = rouge_scores['rouge-l']['p']
    self.results['rouge_l_recall'] = rouge_scores['rouge-l']['r']
    self.results['rouge_l_f1'] = rouge_scores['rouge-l']['f']

  def compute_distinct_n(self, candidates):
    """Computes the diversity of the generated responses.
       dist-n is defined as the ratio of unique n-grams (n=1;2) over all n-grams in the generated responses
    Args:
      candidates (list): list of generated outputs by the model
    
    Returns:
      inter_dist1, inter_dist2: inter-dist as the distinct value among all sampled responses
      intra_dist1, intra_dist2: intra-dist as the average of distinct values within each sampled response
    """
    batch_size = len(candidates)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()

    for seq in candidates:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))
        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    self.results['inter_dist1'] = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    self.results['inter_dist2'] = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    self.results['intra_dist1'] = np.average(intra_dist1)
    self.results['intra_dist2'] = np.average(intra_dist2)

  def compute_semantic_textual_similarity(self, candidates, references):
    """Computes the embeddings of the sequences based on pre-trained sentence transformers and measures the cosine similarity between them
    Args:
      candidates (list): list of generated outputs by the model
      references (list): list of ground-truth sentences
    Returns:
      averaged cosine similarity score
    """
    embeddings1 = self.sentence_transformer.encode(candidates, convert_to_tensor=True)
    embeddings2 = self.sentence_transformer.encode(references, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    total_cosine_score = 0
    for i in range(len(cosine_scores)):
      total_cosine_score = total_cosine_score + cosine_scores[i][i]
    self.results['semantic_textual_similarity'] = total_cosine_score.item()/(len(cosine_scores))
    