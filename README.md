# NER Tagger

This repository contains code for a Named Entity Recognition (NER) tagger. The NER tagger is trained on CoNLL 2003 data and uses a perceptron-based approach.

## Code Overview

The code is organized into several functions:

1. `decode(input_length, tagset, score)`: This function computes the highest scoring sequence according to the scoring function. It takes the input length, tagset, and scoring function as parameters and returns the highest scoring tag sequence.

2. `compute_score(tag_seq, input_length, score)`: This function computes the total score of a tag sequence. It takes the tag sequence, input length, and scoring function as parameters and returns the total score.

3. `compute_features(tag_seq, input_length, features)`: This function computes the feature vector for a given tag sequence. It takes the tag sequence, input length, and feature function as parameters and returns the feature vector.

4. `sgd(training_size, epochs, gradient, parameters, training_observer)`: This function implements stochastic gradient descent. It takes the training size, number of epochs, gradient function, initial parameters, and training observer function as parameters and returns the final parameters.

5. `train(data, feature_names, tagset, epochs)`: This function trains the model on the provided data. It takes the training data, feature names, tagset, and number of epochs as parameters and returns the learned parameters.

6. `predict(inputs, input_len, parameters, feature_names, tagset)`: This function predicts the tags for a given input sequence. It takes the inputs, input length, parameters, feature names, and tagset as parameters and returns the predicted tags.

7. `make_data_point(sent)`: This function creates a dictionary representing a data point from a given input sentence.

8. `read_data(filename)`: This function reads the CoNLL 2003 data from a file and returns an array of dictionaries representing the data.

9. `write_predictions(out_filename, all_inputs, parameters, feature_names, tagset)`: This function writes the predictions on all_inputs to an output file in CoNLL 2003 evaluation format.

10. `evaluate(data, parameters, feature_names, tagset)`: This function evaluates the precision, recall, and F1 score of the tagger compared to the gold standard in the data.

11. `test_decoder()`: This function tests the `decode` function.

12. `main_predict(data_filename, model_filename)`: This is the main function for making predictions. It loads the model file and runs the NER tagger on the data, writing the output to a file.

13. `main_train()`: This is the main function for training the model.

## Usage

To use this code, you can follow these steps:

1. Prepare the training data in CoNLL 2003 format.

2. Call the `train` function to train the model on the training data. Provide the training data, feature names, tagset, and number of epochs as parameters. This function will return the learned parameters.

3. Optionally, you can save the learned parameters to a file using the `write_to_file` method of the `FeatureVector` class.

4. Call the `main_predict` function to make predictions on new data. Provide the data file and the model file as parameters. This function will write the predictions to an output file in CoNLL 2003 evaluation format.

## Dependencies

This code has the following dependencies:

- `random`
- `conlleval`

Please make sure to install these dependencies before running the code.