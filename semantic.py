import spacy
nlp = spacy.load('en_core_web_md')

## Similarity between words ##
#The similarity between "cat" and "monkey" is moderate. This is expected, as they are both animals but from different species.
#The similarity between "banana" and "monkey" is higher, possibly due to the fact that monkeys are often associated with eating bananas.
#The similarity between "banana" and "cat" is relatively low; this is reasonable, as cats and bananas are not closely 
#related in terms of semantic meaning.

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

## My own example of similarities between words ##

#The similarity scores between "house" and "road" are typically relatively low. This outcome is expected since these words belong 
#to different semantic categories and do not share strong connections. 
#The similarity scores between "road" and "juice" are also expected to be low. These words come from entirely different domains 

word_1 = nlp('house')
word_2 = nlp('road')
word_3 = nlp('juice')
print(word_1.similarity(word_2))
print(word_2.similarity(word_3))
print(word_1.similarity(word_3))

#Similarity between words using a loop
tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
   for token2 in tokens:
       print(token1.text, token2.text, token1.similarity(token2))
      

#Similarity between sentences using a for loop
sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
   similarity = nlp(sentence).similarity(model_sentence)
   print(sentence + " - ", similarity)      



#Run the example file with the simpler language model ‘en_core_web_sm’and write a note on what you notice is different from the model
#'en_core_web_md':

#In summary, the most noticeable differences between the en_core_web_sm and en_core_web_md models lie in their technical aspects, 
#such as model size, vector dimensions, and performance on semantic tasks. The en_core_web_sm model is significantly smaller in size 
#compared to en_core_web_md. This size reduction leads to faster loading times and lower memory usage, making it more suitable for systems with limited resources.
#The word vectors in en_core_web_sm is significantly smaller in size compared to 'en_core_web_md'. This decrease in performance is expected since the larger model is 
#trained on more extensive and diverse data.