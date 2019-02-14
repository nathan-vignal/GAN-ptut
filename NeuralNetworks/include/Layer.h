#ifndef LAYER_H
#define LAYER_H
#include "Neuron.h"
#include <vector>


class Layer
{

private:
    std::vector<Neuron *> neurons;
public:
    const std::vector<Neuron *> &getNeurons() const;

public:

    Layer(const unsigned short & nbNeurons,const unsigned short & nbNeuronsInPreviousLayer);
    virtual ~Layer();
    unsigned short getNumberOfNeurons();
    std::vector <std::vector<float>>  getMyactivations();
    void processMyNeuronsActivations(const std::vector <std::vector<float>> & previousLayerActivations);
    void resetActivations();
    void processLastLayerError(std::vector<std::vector<float>> output);
    void processLayerError(const Layer *nextLayer) ;
    std::vector<float> getErrorFromVector();
    std::vector<float> getWeightFromVector();
    std::vector<std::vector<float>> getNeuronErrors()const;
    void layerGradientDescent(const std::vector<std::vector<float>> &previousLayerActivation,
                                  const double &regularizationTerm);
    void addActivations(std::vector<std::vector<float>> newActivations);

    std::vector<std::vector<float>> getNeuronWeight();
    friend std::ostream& operator<< (std::ostream& stream, Layer & layer);
};

#endif // LAYER_H
