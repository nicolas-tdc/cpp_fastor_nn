#include <Fastor/Fastor.h>
#include <vector>

const size_t InputsCount = 10;
const size_t OutputsCount = 5;
const size_t HiddenLayersCount = 20;
const size_t NeuronsPerLayer = 512;

typedef float Scalar;
typedef Fastor::Tensor<Scalar, InputsCount> InputTensor;
typedef Fastor::Tensor<Scalar, OutputsCount> OutputTensor;
typedef Fastor::Tensor<Scalar, NeuronsPerLayer> HiddenTensor;
typedef Fastor::Tensor<Scalar, HiddenLayersCount + 1, NeuronsPerLayer> Matrix;

class NeuralNetwork {

public:
    // Constructor
    NeuralNetwork(std::vector<uint> topology, Scalar learning_rate = Scalar(0.005));

    // Forward propagation.
    void propagateForward(InputTensor& inputs);

    // Backward propagation of gradients.
    void propagateBackward(OutputTensor& outputs);

    // Calculate gradients of each layer.
    void calculateGradients(OutputTensor& outputs);

    // Update weights from gradients.
    void updateWeights();

    // Train the network with input and output data.
    void train(InputTensor inputs, OutputTensor outputs);
    
private:
    std::vector<HiddenTensor> neuronLayers; // Values.
    std::vector<HiddenTensor> cacheLayers; // Unactivated values.
    std::vector<HiddenTensor> gradients; // Gradients.
    std::vector<Matrix*> weights; // Matrix of weights.
    Scalar learningRate; // Learning rate.

// Getters and setters
public:
    void setNeuronLayers(std::vector<HiddenTensor>);
    std::vector<HiddenTensor> getNeuronLayers() { return neuronLayers; };

    void setCacheLayers(std::vector<HiddenTensor>);
    std::vector<HiddenTensor> getCacheLayers() { return cacheLayers; };

    void setGradients(std::vector<HiddenTensor>);
    std::vector<HiddenTensor> getGradients() { return gradients; };

    void setWeights(std::vector<HiddenTensor>);
    std::vector<Matrix*> getWeights() { return weights; };

    void setLearningRate(Scalar);
    Scalar getLearningRate() { return learningRate; };
};