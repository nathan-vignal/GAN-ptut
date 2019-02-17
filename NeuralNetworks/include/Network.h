#ifndef NETWORK_H
#define NETWORK_H

#include <ostream>
#include "Layer.h"

class Network
{

private:
    double regularizationTerm;
    unsigned indexStartDiscriminator;
    std::vector<float> costs;
    std::vector<std::vector<std::vector<float> >> entries;
    std::vector<std::vector<std::vector<float> >> fakeEntries;
    std::vector<std::vector<std::vector<float> >> realEntries;
    std::vector<std::vector<std::vector<float> >> output;
    unsigned numberOfEpochs;
    std::vector<Layer *> layers;





public:
    const static float learningRate ;
    Network(const std::vector<unsigned short> &hiddenLayersArchitecture,
            const std::vector<std::vector<float> > &_entries,
            const std::vector<std::vector<float> > &_output, const unsigned int &_numberOfEpochs,
            const double &_regularizationTerm, const unsigned &_indexStartDiscriminator);
    virtual ~Network();
    const std::vector<float> &getCost() const;
    const std::vector<Layer *> &getLayers() const;
    void main();
    void feedforward(const unsigned int &numberOfTheEpoch, const bool &trainGenerator);
    std::vector<float> testFeedforward(const std::vector<float> &entries, const bool &testGenerator);
    void processCost(const unsigned int &batchNumber);
    long double processMeanError(const unsigned int &numberOfTheEpoch);
    void resetActivations();
    //void vectorResizing(std::vector<std::vector<float>> vector, unsigned rows, unsigned columns );
    void backPropagation(const unsigned int &numberOfTheEpoch, const bool &trainGenerator);
    void gradientDescent(const unsigned int &batchNumber, const bool &trainGenerator);
    friend std::ostream& operator<< (std::ostream& stream, Network & network);
};

#endif // NETWORK_H
