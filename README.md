Simplified version of Adam Vogel and Dan Jurafsky. "Learning to follow navigational directions" https://nlp.stanford.edu/pubs/spatial-acl2010.pdf

Major simplifications:
1) No allocentric or egocentric features in the feature vector
2) Number of iterations instead of theta convergence

Settings:
1) Random action selection or Boltzmann exploration
2) Temperature 2.0 for Boltzmann exploration
3) Alpha discount factor is 10.0 / (10.0 + utteranceNumber / dialogueNumber)
