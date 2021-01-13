class BagOfWords:
    """
    Bag of words quantization class

    Methods that start with an underscore are meant to be used
    internally.

    Usage:

    document = ... # <--- some string/image/data structure/etc.
    words = getWordsOfDocument(document) # <-- some arbitrary method that
                                         #     returns a list of
                                         #     strings/vectors/some other
                                         #     *hashable* arbitrary data
                                         #     structure
    bow = BagOfWords()
    bow.add_document(document, words)

    bow.words # <-- The dictionary
    bow.get_doc_vector(some_word) # <-- Document vector of a single word
    
    """

    def __init__(self):
        self.words = {}
        self.docs = []
        self.doc_vectors = {}
    
    def add_document(self, doc, words_of_doc):
        for word in words_of_doc:
            self._add_word(word, len(self.docs))
        self.docs.append(doc)

    def _add_word(self, _word, doc_id):
        if type(_word) == "list":
            word = tuple(_word)
        word = _word
        if not self.words[word]:
            self.words[word] = {}

        if not self.words[word][doc_id]:
            self.words[word][doc_id] = 1
        else:
            self.words[word][doc_id] += 1
            
        
    def get_doc_vector(self, _word):
        if type(_word) == "list":
            word = tuple(_word)
        word = _word
        if not word in self.doc_vectors:
            self.doc_vectors[word] = []
            for i in range(len(self.docs)):
                self.doc_vectors[word].append(self.words[word][i])
        return self.doc_vectors[word]