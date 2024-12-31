# Desarrollado por "Fernando Alejandre"
# En este caso, la clase "Review", representa un objeto en el archivo json proporcionado.
# El objetivo de crear cada Json en un objeto es poder explorar otros datos como el rating o si la compra fue verificada

class Review:
    rating = 0
    title = ''
    comment = ''
    helpful_vote = 0
    verified = False
    bert_grade = 0

    def __init__(self):
        pass

    def init_from_json(self, json):
        self.rating = json["rating"]
        self.title = json["title"]
        self.comment = json["text"]
        self.helpful_vote = json["helpful_vote"]
        self.verified = json["verified_purchase"]
    
    def set_bert_grade(self, grade):
        self.bert_grade = grade

    def __str__(self):
        if self.bert_grade > 0:
            return "[" + self.comment + "," + str(self.bert_grade) + "]"
        return "[" + self.comment + "]"