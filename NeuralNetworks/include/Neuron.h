#ifndef NEURON_H
#define NEURON_H
#include <vector>
#include <ostream>
#include <random>

class Neuron
{
public:
    static unsigned short maxWeight;
    static unsigned short maxBias;

private:


    std::vector<float> activations;
    std::vector<float> preActivation;
    std::vector<float> weights;
    std::vector<float> errors;
    float bias;

public:
    const std::vector<float> &getErrors() const;
    const std::vector<float> &getPreActivation() const;
    Neuron( const unsigned short & nbWeights );
    virtual ~Neuron();
    void processActivations(const std::vector<std::vector<float>>& previousLayerActivations);
    static float sigmoid(float x);
    static std::vector<float> hadamardProduct(const std::vector<float> & vector1 ,const std::vector<float> & vector2 );
    static float sigmoidPrime(float x);


    //accesseurs
    const std::vector<float> & getActivations() const;
    const float & getActivation(const unsigned short adress );
    std::vector<float> getError();
    std::vector<float> getWeights();
    void debugSetBias(int newBias);

    void addError(float error);
    void gradientDescent(const std::vector<std::vector<float>> &previousLayerActivations, const double &regularizationTerm);

    void resetActivations();
    void processLastNeuronError(const std::vector<float> &outputError);

    friend std::ostream &operator<<(std::ostream &os, const Neuron &neuron);
    void addActivation(const float & newActivation);
};



#endif // NEURON_H
