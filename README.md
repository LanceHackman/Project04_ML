# Machine Learning Project 04: Improved Hexameter AI

This an AI that predict the scansion of a line of Latin Hexameter Poetry.

# Project Overview

The project trains 3 separate ML models: CNN, RNN(LSTM), and Transformer. These models can be run in Main.py, along with a base Macronizer, to answer questions on Hexameter.co. The code is a self-sufficent cycle, because better code gets lines from Hexameter (my source of data) much faster. 

# Instruction to run it

You need several packages installed first. Make sure that they are updated.
 ```bash
```

You will also need to install with homebrew:
Chrome WebDriver: For web automation functionality


Run the programs in this order
1) train a model
2) run Main

Make sure already have an Hexameter.co account before you run step 4

# Reasources used

- Assisted by FlintAI and ClaudeAI to write  of the code. 
- To learn the way
    - https://www.wolframcloud.com/automatic-metrical-scansion-of-latin-poetry-in-dactylic-hexameter--2019-07-5kj8o7i/
    - https://www.ibm.com/think/topics/recurrent-neural-networks#763338458

# Metrical Pattern Encoding

- Hexameter patterns are encoded as 6-character strings:
 - D: Dactyl (— ∪ ∪) - long followed by two shorts
 - S: Spondee (— —) - two long syllables
- Example: DSSSDS = Dactyl-Spondee-Spondee-Spondee-Dactyl-Spondee


# How it Works
- Main.py connects to Hexameter.co. It will run on an infinite loop, answering questions and collecting lines. The user chooses the model used.
- CNN.py builds, trains, saves, and can load the CNN model
- LSTM.py builds, trains, saves, and can load the LSTM model
- Transformer.py builds, trains, saves, and can load the transformer model


# Future Enhancements
- Cleaner interface
- Organize code and files
- More comparisons
- Overcome Alatius
