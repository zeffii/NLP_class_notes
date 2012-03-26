# Natural Language Processing #

Annotations and personal elaborations on the Coursera material for the Natural Language Processing course. No affiliation to the course is implied and my interpretation of the content can be flawed. This document is purely intended to assist me in the process of learning the subject and is a mix between transcript and personal notes.

This Markup is best read in markdownpad from [markdownpad.com](http://www.markdownpad.com "markdownpad"). Sadly the original implementation of markdown does not include support for super and subscript, if it bugs me enough it might be worth writing a parser that does.

### Language modeling ###

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


### Estimating N-gram Probabilities ###


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

  
### Evaluation: How good is our model? ###

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

Perplexity is the ***weighted equivalent branching factor***. Numbers example:  

-  Operator (1 in 4)
-  Sales (1 in 4)
-  Technical Support ( 1 in 4 )
-  30,000 names ( 1 in 120,000 each =  1/4 * 1/4 * 1/4 * 30,000)  
-  Perplexity is: 53 (52.64...)

Lower perplexity indicates a better trained model. 

My python interpretation of the formula is as follows, rewriting it this way helped me understand the whole idea better:

	import math
	
	def perplexity(chances):
	    """pythonification of the formal definition of perplexity.

		input: 	a sequence of chances (any iterable will do)
		output:	perplexity value.
		"""

	    N = len(chances)
	    product = 1
	    for chance in chances:
	        product *= chance
	
	    # return product**(-1/N)
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

### Generalization and zeroes ###

What to do when we see a lot of zeroes? It helps to consider the Shannon visualization method.

-  choose a random bigram
-  now choose a bigram that starts with the previos word
-  go on until you choose the end of sentence token.

looks like:

	< s >	I
		  	I	want
		  		want	to
						to	eat
							eat Chinese
								Chinese	food
										food	< /s >	

**Shakespeare as corpus:**

-  N = 884,647 words (tokens)
-  V = 29,066 vocabulary (unique roots)
-  Produced 300K bigram types out 844 million possible = (V^2)

So 99.96% of the possible bigrams remain unseen and will have 0 in the table. A vast number of zeroes.

-  Quadrigrams: in shakespearean text produce lines that are themselves mostly direct qoutes from shakespeare. Because the corpus(N) is so small. Try it.

**The perils of overfitting**

N-grams only work well for word prediction if the test corpus looks like the training corpus. Language model trained on Shakespeare will have an abysmal result when looking at the Wallstreet Journal. 

> While A model trained on the Wallstreet Journal (corpus A) might do better on another financial publication (corpus B). 

We need to train models that do a better job of generalizing, make them more robust. Combinations that don't occur in the training set will result in zeros, but the test set might validly contain such combinations.

Bigrams with zero probability will result in division by zero when calculating perplexity, so they must be counteracted.

### Bigrams with zero probability ###

How do we deal with these beasts?
Simplest idea:
**Smoothing**.  

Smoothing: Add-one also called Laplace smoothing: _**Add-one estimation**_

> If we have sparse statistics. We want to steal probability mass, for combinations that we might not see later, and place it on combinations that didn't occur in the training data.

in brief:

> **Laplace smoothing**  :  
> Pretend we saw each word one more time that we did.
> Add one to all counts for all bigrams.

The formal expression:   
v = words in sentence.

-  P Add-1(Wi|Wi-1) = C(Wi-1, Wi) + 1  /  C(Wi-1)+V  

With (Maximum Likelyhood Estimate) **MLE** suppose a word occurs 400 times in a corpus of 1 million words. The MLE is 400/1, 000,000 = 0.0004. *This may be a bad estimate for the likelyhood of that word occuring in another corpus*

> PP ML(C) <= PP smoothed(C)  

Add-1 estimation makes a massive difference if you compare the result on the reconstituted bigram table vs the raw bigram table, some differences are up to a factor of 10. So now we got rid of the zeros but gained a much greater level of uncertainty about valid syntax. **There are better methods!**

Add-1 is used to smooth other NLP models  

-  For text classification  
-  In domains where the of zeros isn't so huge.  

It helps for me to think of this as an analogy to the concept of antialiasing in bitmap-graphics.

----------

Further reading on smoothing:  
*Microsoft Research / Stanley F. Chen & Joshua Goodman*
[http://research.microsoft.com/~joshuago/tr-10-98.pdf](http://research.microsoft.com/~joshuago/tr-10-98.pdf "An Empirical Study of Smoothing Techniques for")


----------


## Interpolation ##

Sometimes it helps to use **less** context, applying the requirement for less context with respect to context/situations you haven't learned much about. (rewrite)?

**Backoff:**  

-  _Use a trigram if you have good evidence/data_  
-  What if you haven't seen a trigram, you look at the bigrams, or if they don't exist for that combination, then you can look look at the unigram.

**Interpolation:**  

-  mixing unigram, bigram, trigram. 
-  Interpolation tends to work better than backoff. **(clarify?)**

There are (broadly speaking) two kinds of interpolation.  
**(lambda=λ, must sum to 1 to make them a probability)**

Linear Interpolation: 

- **Simple interpolation**   
  
  Adding 1gram+2gram+3gram together depending on weights (λ)  
  P_hat(Wn|Wn-1Wn-2) =  
  λ1 * P(trigram) + λ2 * P(bigram) + λ3 * P(unigram)  
		
- **Lambdas conditional on context:** (slightly more complicated)  

  Now lambdas are dependant on what the previous two words were.

**Where do the lambdas come from?** How to set lambdas? _We use a held-out corpus_. Choose λ to maximize the probability of held-out data:  

-  Fix the N-gram probabilities (on the training data)
-  Then search for λs that give the largest probability to held-out set.

(LaTeX needed)
log P(Wi...Wn|M(λ1...λk)) = sigma over all i log Pm(λ1....λk) ( Wi|Wi-1)  

The held out corpus can be used to set the lambdas, the idea is we take training data, train some Ngrams then choose which lambdas I would use to interpolate those ngams such that it gives me the highest probability of (predicting accurately) this held out data. Find the set of probabilities such that log probabilities of the words of the held out data are highest.


##Unknown words: Open versus closed vocabulary tasks##

If we know all the words in advanced then Vocabulary V is fixed and we are talking about a **Closed vocabulary task** (menus, predefined scenarios). Often we don't know this (if a vocabulary is fixed or not) in advance. This translates into **Out of Vocabulary** (OOV) words, and turns the exercise into an **Open vocabulary Task**. 

Instead: utilize an unknown word token < UNK > and train on < UNK > probabilities.  

-  Create a fixed lexicon L of size V  
-  At text normalization phase, any training word not in L (all OOV words..or rare words) changed to < UNK >  
-  Now we train its probabilities like any normal word.  
-  At decoding time, if text input (doesn't match) use UNK probabilities for any such word not in the training data. 

**Huge web-scale n-grams**  

How do we deal with computing probs in such large spaces, we prune. we only story non singleton counts. or compute the entropy/plexity and decide to remove ones that don't contribute (entropy based pruning)

**Efficiency**  

-  Efficient data structures like *tries*  
-  Bloom filters: apx language models (clarification needed)
-  Store as indices, not strings.  
	-  use huffman encoding to fit large numbers of words into 2 byes.
-  Quantize probabilities (4-8 bits instead of 8-byte float)

**Smoothing for web-scale N-grams**

Use **"stupid backoff"**? (Brants et al. 2007)  (uses scores, rather than probabilities)
or **No Discounting**, just use relative frequencies.

-		Insert LaTeX here

Add-1 smoothing is ok for text classification but not so great for language modeling. The mot commonly used method of interpolation is the **Extended Kneser-Ney method (elaborate)**. However, for very large N-grams like the web, *stupid backoff* is often adequate.

**Advanced Language Modeling**  

Recent research:

-  **Discrimintive models**: choose n-gram weights to improve a task, not to fit the training set/data.  
-  **Parsing-based models** ( elaborate soon ).    
-  **Caching Models**: train on recently used words, they have increased probability of reappearing.  
Pcache(W|history) =  
lambdaP(Wi|Wi-2, Wi-1) + (1-lambda)(C(W function of history / |history|)    
	- Caching model has weak performance in speech recognition (Why?)

## Good Turing Smoothing ##

-  General Formulations (LaTeX)  
-  Unigram Prior Smoothing (LaTeX), works well, but not great for **LM**)  
-  Advanced smoothing algorithms, based on some form of intuition
	-  Good-Turing  
	-  Kneser-Ney  
	-  Witten-Bell  

>  *The goal of smoothing algorithms is to replace those unseen zeros with a number that relates to a proposed likelyhood of being present in the body of text being looked at. Sometimes this means using the counts of words that you've seen once to help estimate the count of things we've never seen.*

**Notion: N sub c** = Frequency of frequency c  

-  Nc = count of things we've seen c times.  
-  Sam I am I am Sam I do not eat  

	I	3	<	N sub 3 = 1  
	
	sam 2	<	  
	am	2	<	N sub 2 = 2  
	
	do	1	<  
	not	1	<  
	eat	1	<	N sub 1 = 3

Imagine this scenario (by Josh Goodman). You are fishing and catch: 10 carp, 3 perch, 2 whitefish, 1 trout, 1 salmon, 1 eel. A total of 18 fish.  

*How likely is it that the next species is trout?* 1/18  
That's fine for species your encountered previously. But how to respond to a new species, *what is the likelyhood of catfish or bass?*  

-  We can use the estimate of things-we-saw-once to estimate the new things.  
-  3/18 (because N sub 1 = 3)  

Assuming this statement, how likely is it that the next species is trout? It must be less than 1/18 because are going to use probability mass from each fish we have seen and add it to the unseen fish probability. **How do we adjust the new probability spread?**  
  
**Good Turing calculations**  
>  P star subGT (things with zero frequency) = N1 / N  
  
>  c star = ((c+1)*Nsubc+1) / Nsubc  

-  Unseen fish ( bass, catfish) 
	- c = 0;
	- MLEp = 0/18 = 0

	adjusted for good turing:  
	- P star sub GT(unseen) = Nsub1/N = 3/18  

- Seen fish once (trout)  
	- c = 1
	- MLE p = 1/18  

	adjusted for good turing:  
	- c star(trout) = (2*Nsub2) / Nsub1 = (2*1)/3 = 2/3  
	- P star sub GT(trout) = (2/3) / 18 = 1/27  

**Complications of Good-Turing**  

> we can't always use N sub k+1, 

- For small k, Nsubk > N subk+1
- For large k, too jumpy, zeros wreck estimates.

- Simple Good-Turning [Gale & Sampson] replace empircal Nsubk with best-fit power law once count counts get unreliable. (use a power log function to perform best fit after a certain count of Nsubk) (The graph them combines discreet histogram and at a cutoff point converts everything to a long tail curve)

**Resulting good-Turing numbers:**  
Example: Numbes from Church and Gale (1991) / 22 million words of AP newswire.

c* = {{(c+1)N_{c+1}} \over {N_c}}  

What is the general relationship betwen these counts?

	Count c |	Good Turing c*  
	0		|	0.0000270  
	1		|	0.446  
	2		|	1.26  
	3		|	2.24  
	4		|	3.24  
	5		|	4.22  
	6		|	4.19  
	7		|	6.21  
	8		|	7.24  
	9		|	8.25  

Mostly here it's -.75 on each count.

## Kneser-Ney ##

One of the most sophisticated, is Kneser-Ney and it might be intuitive.  

**Absolute Discounting Interpolation**  

- save time by doing no brain subtraction of .75 (or some other discount value for different corpera)  
P sub AbsoluteDiscounting(Wi|Wi-1) =  
(c(Wi-1,Wi) - d) / c(Wi-1) + lambda * (Wi-1) * P(W)

**Kneser-Ney Smoothing 1** offers a better estimate for probabilities of lower-order unigram.  
Example from shannon game: <i>I can't see without my reading _________</i>.  
The unigram is useful exactly when we haven't seen a bigram.  

- Instead of P(W): "How likely is w"
- Psub continuation(W): "How likely is w to appear as a novel continuation?"  

>  How do we measure novel continuation? : For each word, count the number of bigram types it completes/creates
>  Every bigram type was a novel continuation the first time it was seen.  
>  		**Pcontinuation(W) is proportional to | { Wi-1:c(Wi-1,W)>0 } |**  

**Kneser-Ney Smoothing 2**
How many times does w appear as a novel continuation:  

-  Pcontinuation(W) \propto | { Wi-1:c(Wi-1,W)>0 } |

Normalized by total number of word bigram types:

- | { (Wj-1,Wj):c(Wj-1,Wj)>0 } |    what is the cardinality of this set,

Pcontinuation(W) = |{Wi-1:c(Wi-1,W)>0}| / |{ (Wj-1,Wj):c(Wj-1,Wj)>0}|  
or LaTeX

  
	P_{CONTINUATION}(w) = {|\{w_{i-1}:c(w_{i-1},w)>0\}| 
	 \over
	|\{(w_{j-1},w_j):c(w_{j-1},w_j)>0\}|}

![p_continuation](http://dl.dropbox.com/u/3397495/Pcontinuation.gif)


  



	
















----------
harvesting semantic relation / espresso http://www.patrickpantel.com/download/papers/2006/acl06-01.pdf