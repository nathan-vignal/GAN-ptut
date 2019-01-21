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
void Layer::processLastLayerError(std::vector<std::vector<float>> output){


    for(unsigned neuronNumber=0; neuronNumber<neurons.size()-1;++neuronNumber){
        std::vector<float> outputForTheNeuronN;
        //modèle le vecteur des valeurs attendu pour ce neuron, à partir du vecteur output
        for(unsigned feedForwardNumber=0; feedForwardNumber<output.size()-1; ++feedForwardNumber){
            outputForTheNeuronN.emplace_back(output[feedForwardNumber][neuronNumber]);
        }
        //donne au neuron le vecteur des valeurs attendu pour qu'il calcule son erreur
        neurons[neuronNumber]->processLastNeuronError(outputForTheNeuronN);
        outputForTheNeuronN.clear();
    }

}
void Layer::processLayerError(Layer nextLayer) {

    float error;

    for (unsigned i(0); i < getErrorFromVector().size() -1; ++i) {
       /*for (unsigned j(0); j < nextLayer.getWeightFromVector().size()-1; ++j) {
           // error += nextLayer.getWeightFromVector()[j] * nextLayer.getErrorFromVector()[j];
        }*/
        //getErrorFromVector()[i] = error * Neuron::sigmoidPrime(neurons[i]->getPreActivation()[i]);
    }

}


std::vector<float> Layer::getErrorFromVector() {
    std::vector<float> errorFromVector;
    for (auto e : this->getNeuronErrors()) {
        for (unsigned i(0); i < e.size() - 1; ++i) {
            errorFromVector.emplace_back(e[i]);
        }
    }
    return errorFromVector;
}

std::vector<std::vector<float>> Layer::getNeuronWeight() {
    std::vector<std::vector<float>> neuronsWeight;
    for (auto neuron : neurons) {
        neuronsWeight.emplace_back(neuron->getPreActivation());
    }
    return neuronsWeight;
}










void Layer::layerGradientDescent(std::vector<std::vector<float>> previousLayerActivation) {
    for(auto neuron : neurons){
        neuron->gradientDescent(previousLayerActivation);
    }


}


std::vector<std::vector<float>> Layer::getNeuronErrors(){
    std::vector<std::vector<float>> neuronsError;
    for(auto neuron : neurons){
        neuronsError.emplace_back(neuron->getError());
    }
    return neuronsError;

}
std::vector<float> Layer::getWeightFromVector() {
    std::vector<float> weightFromVector;
    for(auto w : this->getNeuronWeight()) {
        for (unsigned i(0); i < w.size() -1; ++i) {
            weightFromVector.emplace_back(w[i]);
        }
    }
    return weightFromVector;
}
