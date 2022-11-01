# EOSとは End Of Sentence の略であり，文の終わりを意味する
# EOSの単語IDを0と定義する
EOS_ID = 0
UNKNOWN_WORD_ID = 1

class SentenceData:
    def __init__(self, file_name):
        with open(file_name, "r") as f:
            self.en_word_to_id = {"<EOS>": EOS_ID, "<UNKNOWN>": UNKNOWN_WORD_ID}
            self.en_word_list = ["<EOS>", "<UNKNOWN>"]
            self.jp_word_to_id = {"<EOS>": EOS_ID, "<UNKNOWN>": UNKNOWN_WORD_ID}
            self.jp_word_list = ["<EOS>", "<UNKNOWN>"]
            self.en_sentences = []
            self.jp_sentences = []
            line = f.readline().rstrip("\n")
            while line:
                sentences = line.split("\t")
                english = sentences[0].split(" ")
                japanese = sentences[1].split(" ")

                # 単語IDのリスト
                en_sentence = []
                for word in english:
                    word = word.lower()
                    id = 0
                    if word in self.en_word_to_id:
                        id = self.en_word_to_id[word]
                    else:
                        id = len(self.en_word_list)
                        self.en_word_list.append(word)
                        self.en_word_to_id[word] = id
                    en_sentence.append(id)

                # 単語IDのリスト
                jp_sentence = []
                for word in japanese:
                    id = 0
                    if word in self.jp_word_to_id:
                        id = self.jp_word_to_id[word]
                    else:
                        id = len(self.jp_word_list)
                        self.jp_word_list.append(word)
                        self.jp_word_to_id[word] = id
                    jp_sentence.append(id)
                self.en_sentences.append(en_sentence)
                self.jp_sentences.append(jp_sentence)
                line = f.readline().rstrip("\n")

    def sentences_size(self):
        return len(self.en_sentences)

    def japanese_word_size(self):
        return len(self.jp_word_list)

    def english_word_size(self):
        return len(self.en_word_list)

    def japanese_sentences(self):
        return self.jp_sentences

    def english_sentences(self):
        return self.en_sentences

    def japanese_word_id(self, word):
        if word in self.jp_word_to_id:
            return self.jp_word_to_id[word]
        else:
            return UNKNOWN_WORD_ID

    def english_word_id(self, word):
        if word in self.en_word_to_id:
            return self.en_word_to_id[word]
        else:
            return UNKNOWN_WORD_ID

    def japanese_word(self, id):
        return self.jp_word_list[id]

    def english_word(self, id):
        return self.en_word_list[id]
