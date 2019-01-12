### Boltzmann Machine for Recommendation System
Contains a class with three functions for the Restricted Boltzmann
Machine which it will obey.
        
1. Initialise tensors of all weights and biases of the  visible nodes and
   hidden nodes. Add weight parameter of the probabilities of the visible 
   nodes according to the hidden nodes.
2. Sample hidden nodes
   For every each hidden node activate them for a given probablity given v.
   In which the activation is a linear function of the neurons where the 
   coefficients are the functions. So, the activation is probability that the
   hidden node will be activated according to the value of the visible node. 
   The activation is returned as a sigmoid function. But we're making a 
   Bernoulli RBM. p[h|v] is vector of nh elements, each element corresponds to 
   each hidden node. We use this probabilities to sample activation of each 
   hidden node, depending on p[h|v] for v. 
3. Sample visible nodes.
   Obtain vector with a binary outcome 
   to list which hidden nodes activated or not activated.
