**Language modeling** is about _**assigning probabilities to a sentence**_.
Used in _machine translation_ and _spell correction_

Goal of language model is (Also called '**the grammar**' or '**LM**':  

- compute probability of a sentence or sequence of words.  
P(W) = P(W1, W2...Wn)  
- Probability of an upcoming word is P(W5|W1, W2...W4)

**How to compute this?** Use the chain rule of probability. Definition of conditional probabilities:  

- P(A|B) = P(A, B) / P(B)
- P(A|B) * P(B) = P(A,B)
- P(A,B) = P(A|B) * P(B)  

To generalize to multiple variables:

- P(A,B,C,D) = P(A) * P(B|A) * P(C|A,B) * P(D|A,B,C)
- P(X1,X2...Xn) = P(X1) * P(X2|X1) * P(X3|X1, X2) * P(Xn|X1,X2..Xn-1)
 






