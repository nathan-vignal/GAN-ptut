#include <iostream>
#include "Layer.h"
Layer::Layer(const unsigned short &nbNeurons, const unsigned short &nbNeuronsInPreviousLayer) {

    for(unsigned i = 0; i<nbNeurons;++i){
        auto * neuron = new Neuron(nbNeuronsInPreviousLayer);
        neurons.emplace_back( neuron );
    }

}

Layer::~Layer()
{
    //dtor
}

std::vector <std::vector<float>>  Layer::getMyactivations() {
    std::vector <std::vector<float>> result(neurons[0]->getActivations().size());
    for(unsigned short i = 0; i< neurons[0]->getActivations().size(); ++i){
        for(Neuron * neuron: neurons){
            result[i].emplace_back(neuron->getActivation(i));

        }
    }
    return result;
}



void Layer::processMyNeuronsActivations(const std::vector<std::vector<float>> &previousLayerActivations) {
    for(auto neuron: neurons){

        neuron->processActivations(previousLayerActivations);
    }

}


std::ostream& operator<<(std::ostream &stream, Layer &layer) {
    for (Neuron *neuron : layer.neurons) {

        stream <<  * neuron;
    }


    return stream;
}

unsigned short Layer::getNumberOfNeurons() {
    return (unsigned short)neurons.size();

}
void Layer::resetActivations() {
    for(auto neuron : neurons ){
        neuron->resetActivations();
    }
}
/**
 * process the derivative of c with respect to z for each neuron and for each feedforward
 * @param output expected output of the feedforward
 */
void Layer::processLastLayerError(std::vector<std::vector<float>> output, const bool &trainGenerator) {

    //pour chaque neurone
    for(unsigned neuronNumber=0; neuronNumber<neurons.size();++neuronNumber){
        std::vector<float> outputForTheNeuronN;
        //modèle le vecteur des valeurs attendu pour ce neuron, à partir du vecteur output
        for(unsigned feedForwardNumber=0; feedForwardNumber<output.size(); ++feedForwardNumber){
            outputForTheNeuronN.emplace_back(output[feedForwardNumber][neuronNumber]);
        }
        //donne au neuron le vecteur des valeurs attendu pour qu'il calcule son erreur à chaque feedForward
        neurons[neuronNumber]->processLastNeuronError(outputForTheNeuronN, trainGenerator);

    }

}

/**
 * calcul l'erreur pour chaque neurone pour chaque feedforward
 * @param nextLayer
 */
void Layer::processLayerError(const Layer *nextLayer)  {


    //pour chaque neurone
    for(unsigned neuronNumber = 0; neuronNumber < neurons.size(); ++neuronNumber){
        float sum = 0;

        //pour erreurs dans le neuron suivants
        unsigned int numberOfFeedFowards = (unsigned int) nextLayer->getNeurons()[0]->getErrors().size();

        for(unsigned feedforwardNumber=0 ; feedforwardNumber < numberOfFeedFowards; ++feedforwardNumber){
            //pour chaque neurone dans le layer suivant
            double numberOfNeuronNextLayer = nextLayer->getNeurons().size();
            for(unsigned nextLayerNeuronNumber = 0 ; nextLayerNeuronNumber < numberOfNeuronNextLayer ;++nextLayerNeuronNumber ){
                Neuron * neuronLinkedByWeight = nextLayer->getNeurons()[nextLayerNeuronNumber];
                sum += neuronLinkedByWeight->getErrors()[feedforwardNumber] *
                        neuronLinkedByWeight->getWeights()[neuronNumber];

            }

            neurons[neuronNumber]->addError(
                    Neuron::activationFunctionPrime(neurons[neuronNumber]->getPreActivation()[feedforwardNumber])* sum);
            sum = 0;


        }

    }
}



void Layer::layerGradientDescent(const std::vector<std::vector<float>> &previousLayerActivation,
                                 const double &regularizationTerm, const bool &trainingGenerator) {
    for(auto neuron : neurons){
        neuron->gradientDescent(previousLayerActivation, regularizationTerm, trainingGenerator);
    }


}


std::vector<std::vector<float>> Layer::getNeuronErrors()const{
    std::vector<std::vector<float>> neuronsError;
    for(auto neuron : neurons){
        neuronsError.emplace_back(neuron->getErrors());
    }
    return neuronsError;

}

const std::vector<Neuron *> &Layer::getNeurons() const {
    return neurons;
}

void Layer::addActivations(std::vector<std::vector<float>> newActivations) {
    for(auto layerActivations: newActivations){
        for(unsigned short neuronNumber = 0; neuronNumber < layerActivations.size();++neuronNumber){
            neurons[neuronNumber]->addActivation(layerActivations[neuronNumber]);
        }



    }


}
