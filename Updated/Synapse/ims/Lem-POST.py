import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


reading = open('anger_plain.txt',errors = 'ignore')
writing = open('anger_plain_parsed.txt','w')

lemmatiser = WordNetLemmatizer()

data_read = reading.read().split('\n')
#print (data_read[0])

for i in range(len(data_read)):
    words = data_read[i].split()
    post = pos_tag(words)
    strin = ''
    for i,word in enumerate(words):
        strin += str(lemmatiser.lemmatize(word)) + ' ' + str(post[i][1]) + '\n'
        
    strin += '\n'
    writing.write(strin)

        
reading.close()
#writing.close()
