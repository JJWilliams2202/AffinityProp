# AffinityProp
This python code was used to cluster words into groups based on similar features. It is an unsupervised machine learning algorithm.
Words do _not_ need to be of the same length to be compared. \
\
The original data included words that were encoded as "word:anotherword" (sans quotations). This code drops all characters before the colon so it only analyses the second word. The words in the original dataset sometimes included special characters and numbers as well as duplicates. For better accuracy, these are removed from the dataset. \
\
Sometimes, when using a large dataset where there are not many shared features between the words, the algorithm will not converge. If this happens, you may need to increase the iterations from the default 1000. For example, a sample of 2000 took 1400 iterations to converge.
