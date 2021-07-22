from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
import spacy
from spacy_sentiws import spaCySentiWS
import sys
nlp = spacy.load('de')
sentiws = spaCySentiWS('/content/drive/My Drive/Colab Notebooks/opinion-lab-group-2.3/refactorization/sentiws')

nlp.add_pipe(sentiws)

lemmatizer = WordNetLemmatizer()
def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def sentence_sentiment(raw_sentence,language= 'en',verbose =False):
  sentiment = 0
  tokens_count = 0
  if language == 'en':
    tagged_sentence = pos_tag(word_tokenize(raw_sentence))
    for word, tag in tagged_sentence:
        wn_tag = penn_to_wn(tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV,wn.VERB):
            continue
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            continue

        synsets = wn.synsets(lemma, pos=wn_tag)
        if verbose:
          print('synsets:', synsets)
        if not synsets:
            continue

        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        
        word_sent = swn_synset.pos_score() - swn_synset.neg_score()
        if verbose:
          print('swn_synset:', swn_synset)
          print('word_sent:', word_sent)
        if word_sent != 0:
            sentiment += word_sent
            tokens_count += 1
    if verbose:
        print('tokens_count: ',tokens_count)
        print('tagged_sentence: ',tagged_sentence)
    if tokens_count == 0:
        return 0
    sentiment = sentiment/tokens_count
    return sentiment
  if language == 'de':
    tagged_sentence = nlp(raw_sentence)
    for token in tagged_sentence:
      if token.pos_ not in ('NOUN', 'ADJ', 'ADV','VERB'):
        continue
      if token._.sentiws != None:
        sentiment += token._.sentiws
        tokens_count += 1
    if tokens_count == 0:
        return 0
    sentiment = sentiment/tokens_count
    return sentiment

