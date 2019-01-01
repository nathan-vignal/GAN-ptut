#ifndef LAYER_H
#define LAYER_H
#include "Neuron.h"
#include <vector>


class Layer
{

private:
    std::vector<Neuron *> neurons;
public:

    Layer(const unsigned short & nbNeurons,const unsigned short & nbNeuronsInPreviousLayer);
    virtual ~Layer();
    unsigned short getNumberOfNeurons();
    std::vector <std::vector<float>>  getMyactivations();
    std::vector<float> getTheLastBatchActivation();
    void processMyNeuronsActivations(const std::vector <std::vector<float>> & previousLayerActivations);
    friend std::ostream& operator<< (std::ostream& stream, Layer & layer);

};

#endif // LAYER_H
