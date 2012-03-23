# Natural Language Processing #

Annotations and personal elaborations on the Coursera material for the Natural Language Processing course. No affiliation and intended purely to assist my learning the subject.


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



