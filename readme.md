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

Zeroes in the probability matrix arise because a corpus doesn't contain certain word combinations. That doesn't mean the word could never follow that word in general english. A zero can indicate that certain words don't logically follow another word, this is especially more visible with much larger bodies of mixed content text.  

**log space** 
With an abundance of near zero values, arithmetic is made faster by using additions and logs.  

> "Avoids arithmetic underflow"

-  p1 * p2 * p3 * p4 = logP1 + logP2 + logP3 + logP4

Publicly available language modeling toolkits:  

-  srilm / speech.sri.com    
-  google N-gram release  (pgoogle web corpus)
-  google books n-grams (http://ngrams.googlelabs.com/

  
## Evaluation: How good is our model? ##

We want to assign higher probability to 'real' or 'frequently observed' sentences and assign lower probability to 'ungrammatical' or 'rarely observed' sentences.

We train on a **training set** and test the performance of the resulting **LM** on a **test set**. Test sets are never before seen virgin data. An **evaluation metric** tells us how well our model does (responds) on the test set.

**Extrinsic evaluation of N-gram models**

The best way to evaluate how two models (A and B) compare is to apply each model to the same task. Tasks like spelling corrector/speech recognizer/Machine Translation System (MT). Run the tasks and analyze the accuracy of A and B, how many mispelled words corrected, how many words recognized correctly or how many words translated correctly.

There is a problem with this analysis method given the Extrinsic **(in vivo)** nature of N-gram models. **It is time/processing expensive** (days/weeks)

Sometimes another way to evaluate models is by **intrinsic evaluation** called **perplexity**, but it is a poor/bad approximation if the test and training data don't share a lot of similarity. The assertion is that perplexity is ok if two data sets are very similar and that can be OK for **pilot experiments.** Both are valuable methods.

**intuition of Perplexity**

Claud Shannon. The Shannon Game: how well can we predict the next word? A good language model will attempt to look at the context to narrow down the body of probable words. **uni-grams suck at this game**. 

> If you can guess the next word right, then you are a good language model

**perplexity**, the best language model is one that best predicts an unseen test set. It assigns/gives the highest probability to the sentences that it sees. More formally:

-  *perplexity* is the probabilty of the test set, normlized by the number of words.  
- PP(W) = P(W1, W2... Wn) ^ (1/n)
- 		= N root ( 1 / P(w1,W2...Wn) )

When chain ruled:  

-  PP(W) = N root ( product N over all I,  1 / P(Wi|W1...Wi-1) )  

When chain ruled (bigram only):  

-  PP(W) = N root ( product N over all I,  1 / P(Wi|Wi-1) )

> "minimizing perplexity is the same as maximizing probability"


**Second idea on perplexity** comes from **Josh Goodman**, based on Shannon :  

>  How hard is the task of recognizing digits '0,1,...9' ? 

Perplexity is related to **Average branching factor**, on average how many things could come next at any point in sentence. (it's related to the entropy on the up comming things)

Example given: Speech recognition for automated operator to connect to an employee. 30,000 unique full names gives a perplexity of 30,000.

Perplexity is the weighted equivalent branching factor. Numbers example:  

-  Operator (1 in 4)
-  Sales (1 in 4)
-  Technical Support ( 1 in 4 )
-  30,000 names ( 1 in 120,000 each =  1/4 * 1/4 * 1/4 * 30,000)  
-  Perplexity is: 53 (52.64...)

Lower perplexity indicates a better trained model. 

My python interpretation of the formula is as follows, rewriting it this way helped me understand the whole idea better:

	import math
	
	def perplexity(chances):
	    '''
	    pythonification of the formal definition of perplexity.
	    '''
	    N = len(chances)
	    product = 1
	    for chance in chances:
	        product *= chance
	
	    # return product**(1/N)
	    return math.pow(product, -1/N)
	
This is the general gist of the formula, I would run tests on variations and optimize for speed instead of readability (especially when considering larger sequences). For instance the product of chances might better be calculated like:

	# http://choorucode.wordpress.com/2010/02/05/python-product-of-elements-of-a-list/

	import functools
	import operator
	
	def product(seq):
	    """Product of a sequence."""
	    return functools.reduce(operator.mul, seq, 1)
	


----------

*My net searches on the subject of perplexity:*

>  Gregor Heinrich: http://www.arbylon.net/publications/text-est.pdf 


----------

