#pragma once

#include <Fastor/Fastor.h>
#include <vector>

#include "Layers/InputLayer.h"
#include "Layers/HiddenLayer.h"
#include "Layers/OutputLayer.h"

class NeuralNetwork {

public:

    // Constructor and destructor.

    NeuralNetwork(Scalar learning_rate = Scalar(0.005));

    ~NeuralNetwork();

    // Processing.

    // Forward propagation.
    void forward(BaseTensor& inputs);

    // Backward propagation.
    void backward(BaseTensor& outputs);

    // Update weights from error.
    void update_weights();

    // Train the network with input and output data.
    void train(BaseTensor& inputs, BaseTensor& outputs);

    // Getters and setters.

    // Layers.
    void set_layers(Fastor::Tensor<Scalar> architecture);
    std::vector<BaseLayer> get_layers();

    // Learning rate.
    void set_learning_rate(Scalar);
    Scalar get_learning_rate() { return learning_rate; };

private:

    // Properties.

    std::vector<BaseLayer> layers;

    Scalar learning_rate;

};