# Natural Language Processing #

Annotations and personal elaborations on the Coursera material for the Natural Language Processing course. No affiliation and intended purely to assist my learning the subject.
This Markup is best read in markdownpad from markdownpad.com


**Language modeling** is about _**assigning probabilities to a sentence**_.
Used in _machine translation_ and _spell correction_

Goal of language model is (Also called '**the grammar**' or '**LM**'):  

- compute probability of a sentence or sequence of words.  
P(W) = P(W1, W2...Wn)  
- Probability of an upcoming word is P(W5|W1, W2...W4)

**How to compute this?** Use the chain rule of probability. Definition of conditional probabilities:  

- P(A|B) = P(A, B) / P(B)
- P(A|B) * P(B) = P(A, B)
- P(A, B) = P(A|B) * P(B)  

To generalize to multiple variables:

- P(A, B, C, D) = P(A) * P(B|A) * P(C|A, B) * P(D|A, B, C)
- P(X1, X2...Xn) = P(X1) * P(X2|X1) * P(X3|X1, X2) * P(Xn|X1, X2..Xn-1)
 
**Joint probability** is used for computing the prability of a string of words.  

- sentence = "its water is so transparent"
- P(sentence) = P(its) * P(water|its) * P(is|its water) * P(so|its water is) * 
P(transparent|its water is so)
- P(W1, W2....Wn) = Product over all i, P(Wi|W1, W2...Wi-1)

**Estimate these probabilities?** Yes, the Markov Assumption.  

- sentence = "its water is so transparent that"  
- P("the"|sentence) ≈ P(the|transparent that) 

Formal definition: the markov assumption of the probability of a sequence of words is the product of the conditional probability of the word given some prefix of the last K words.

-  P(W1, W2...Wn) ≈ Product(i) P(Wi|Wi-k...Wi-1)  
-  P(W1|W1W2...Wi-1) ≈ P(Wi|Wi-k..Wi-1)   

The simplest case is **Unigram** model. They are no more than a concatenation of words picked randomly from a body of text.
Unigrams tend to be unintelligable.  K = 0

The **Bigram model** is conditioned on the previous word. K = 1

N-gram uses N=K. While sentences will start to look more like a natural language they will still be insufficient as a model of language because language has **long-distance dependancies**. In most cases the output will be intelligable in at a glance for short stretches of text.

This next line shows that computer and crashed can be separated by a large number of words.  Statistically the likelyhood of the word _crashed_ following the word _floor_ is not high, but it does become high as the subject of the sentence is the computer.  

> "The computer which I had just put into the machine room on the fifth floor crashed"


## Estimating N-gram Probabilities ##

The maximum Likelihood Estimate **MLE**:

-  P(Wi|Wi-1) = count(Wi-1, Wi) / count(Wi-1)

_Or stated in English:_  
Of all the times we saw Wi-1, how many times was it followed by Wi
  
< s > means start symbol  
< /s > means end symbol  

_mr Zeus_ used as a corpus:  
< s > I am Sam  < / s >  
< s > Sam I am < / s >  
< s > I do not like green eggs and ham < / s >  
  
-  P(I|< s >) = 2/3    
-  P(< / s >|Sam) = 1/2  
-  P(Sam|< s >) = 1/3  
-  P(Sam|am) = 1/2     
**Sam** occurs after **am** 1 time, but **am** occurs 2 times in total.  
-  P(am|I) = 2/3  
**am** occurs after **I** 2 times , but **I** occurs 3 times in total.
-  P(do|I) = 1/3  

**Bigram estimates of sentence probabilities**  

P(< s > I want english food < / s >)  
= P(I|< s >) * P(want|I) * P(english|want) * P(food|english) * P(< / s >|food)  
= 0.000031  

Zeroes in the probability matrix arise because a corpus doesn't contain certain word combinations. That doesn't mean the word could never follow that word in general english. A zero can indicate that certain words don't logically follow another word, this is especially more visible with much much largers bodies of mixed content text.   
