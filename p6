import nltk
from nltk.corpus import brown, inaugural, reuters, udhr
from nltk import FreqDist, ConditionalFreqDist
from nltk.tag import UnigramTagger
from nltk.tokenize import word_tokenize
from nltk.corpus import PlaintextCorpusReader

nltk.download('brown')
nltk.download('inaugural')
nltk.download('reuters')
nltk.download('udhr')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('universal_tagset')

print("\nBrown Corpus Categories:", brown.categories())
print("Brown Corpus Sample:", brown.words(categories='news')[:10])

print("\nReuters Corpus Categories:", reuters.categories())
print("Reuters Corpus Sample:", reuters.words(categories='trade')[:10])

print("\nUDHR Languages:", udhr.fileids()[:5])
print("UDHR English Sample:", udhr.words('English-Latin1')[:10])

cfd = ConditionalFreqDist(
    (genre, word.lower())
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)
print("\nMost common words in 'news' category:", cfd["news"].most_common(10))

print("\nTagged Words Sample (Brown Corpus):", brown.tagged_words()[:10])
print("\nTagged Sentences Sample:", brown.tagged_sents(categories='news')[:2])

tagged_words = brown.tagged_words(tagset='universal')
nouns = [word for word, tag in tagged_words if tag == "NOUN"]
fdist = FreqDist(nouns)
print("\nMost Frequent Nouns:", fdist.most_common(10))

word_properties = {
    "run": {"POS": "verb", "Tense": "present", "Meaning": "move swiftly"},
    "apple": {"POS": "noun", "Category": "fruit"},
}
print("\nProperties of 'run':", word_properties["run"])
print("Properties of 'apple':", word_properties["apple"])

patterns = [
    (r'.*ing$', 'VBG'),
    (r'.*ed$', 'VBD'),
    (r'.*es$', 'VBZ'),
    (r'.*ly$', 'RB'),
    (r'.*s$', 'NNS'),
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
    (r'.*', 'NN')
]
regexp_tagger = nltk.RegexpTagger(patterns)
print("\nRule-Based Tagging:", regexp_tagger.tag(["running", "apples", "quickly", "finished", "123"]))

train_sents = brown.tagged_sents(categories='news')[:500]
unigram_tagger = UnigramTagger(train_sents)
print("\nUnigram Tagger Output:", unigram_tagger.tag(["The", "dog", "barks"]))

def segment_text(text, corpus_words):
    words = set(corpus_words)
    segmented = []
    current_word = ""
    for char in text:
        current_word += char
        if current_word in words:
            segmented.append(current_word)
            current_word = ""
    return segmented if segmented else ["No match found"]

corpus_words = set(brown.words())
input_text = "thisisatest"
segmented_words = segment_text(input_text, corpus_words)
print("\nSegmented Words:", segmented_words)
