These are the improvements that have been done to the exercise.

1. ("random-sampling-dtl")
   The random sampling decision tree learner overrides the "get_most_important_attribute" function. It randomly samples
   the original attributes and return a set of sample attributes with the size of int(np.sqrt(len(attributes))).
   This largely improves the time efficiency in finding the highest attribute, as the size of the sampling set has been
   decreased.

2. ("same-sample-efficient-dtl")
   This same sample efficient decision tree learner overrides the "same_target" function. Instead of checking whether
   all the examples in a branch have the same target, we stop iterating when there are more than 20% of the examples 
   have the different targets, which return False. This largely improves the 
   time efficiency, especially when the size of the dataset is very large. Iterating the first 20% of a large dataset
   and making a decision is much more effective than iterating the whole set.