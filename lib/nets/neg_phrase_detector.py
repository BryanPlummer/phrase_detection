import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn


class NPD():
    """
    Negative phrase detector to detect related and exclusive words of a given query word
    """

    def __init__(self, word):
        self.word = word
        self.syn_all = wn.synsets(word)
        self.hypernym = []
        self.hyponym = []
        self.entailment = []
        self.meronym = []
        self.holonym = []
        self.related_words = set()
        self.exclusive_words = set()

    def get_related_words(self):
        """
        find related word by first finding all related synsets,
        then extract its corresponding lemmas
        :return:
        """
        for syn in self.syn_all:
            self.hypernym += syn.hypernyms()
            self.hyponym += syn.hyponyms()
            self.entailment += syn.entailments()

        # get corresponding words
        for syn in self.syn_all:
            for l in syn.lemmas():
                self.related_words.add(l.name())
        for syn in self.hypernym:
            for l in syn.lemmas():
                self.related_words.add(l.name())
        for syn in self.hyponym:
            for l in syn.lemmas():
                self.related_words.add(l.name())
        for syn in self.entailment:
            for l in syn.lemmas():
                self.related_words.add(l.name())

        return self.related_words

    def get_exclusive_words(self):
        """
        find related word by first finding all exclusive synsets,
        then extract its corresponding lemmas
        :return:
        """
        for syn in self.syn_all:
            self.meronym += syn.part_meronyms()
            self.meronym += syn.substance_meronyms()
            self.holonym += syn.part_holonyms()
            self.holonym += syn.substance_holonyms()

        # get all antonyms
        for syn in self.syn_all:
            for l in syn.lemmas():
                for ant_l in l.antonyms():
                    self.exclusive_words.add(ant_l.name())

        for syn in self.meronym:
            for l in syn.lemmas():
                self.exclusive_words.add(l.name())
        for syn in self.holonym:
            for l in syn.lemmas():
                self.exclusive_words.add(l.name())

        return self.exclusive_words


# examples to illustrate the functionality
if __name__ == '__main__':
    query_word = 'tree'
    detector = NPD(query_word)

    detector.get_related_words()
    print("num of related words: {}".format(len(detector.related_words)))
    print(detector.related_words)

    detector.get_exclusive_words()
    print("num of exclusive words: {}".format(len(detector.exclusive_words)))
    print(detector.exclusive_words)

    print('\n')
    query_word = 'engine'
    detector = NPD(query_word)

    detector.get_related_words()
    print("num of related words: {}".format(len(detector.related_words)))
    print(detector.related_words)

    detector.get_exclusive_words()
    print("num of exclusive words: {}".format(len(detector.exclusive_words)))
    print(detector.exclusive_words)

    print('\n')
    query_word = 'drink'
    detector = NPD(query_word)

    detector.get_related_words()
    print("num of related words: {}".format(len(detector.related_words)))
    print(detector.related_words)

    detector.get_exclusive_words()
    print("num of exclusive words: {}".format(len(detector.exclusive_words)))
    print(detector.exclusive_words)

    print('\n')
    query_word = 'chair'
    detector = NPD(query_word)

    detector.get_related_words()
    print("num of related words: {}".format(len(detector.related_words)))
    print(detector.related_words)

    detector.get_exclusive_words()
    print("num of exclusive words: {}".format(len(detector.exclusive_words)))
    print(detector.exclusive_words)