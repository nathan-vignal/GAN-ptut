#include "Neuron.h"
#include <random>
#include <iostream>
#include <Network.h>
//int Exemple::compteur = 0;

unsigned short Neuron::maxWeight = 4;
unsigned short Neuron::maxBias = 4;

Neuron::Neuron( const unsigned short & nbWeights )
{
    /* initialisation des poids est des biais*/
    for(unsigned i =0;i<nbWeights;++i){
        if(rand()%2){
            weights.emplace_back(static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/float(maxWeight))) );//-1 *rand() % maxWeight);
        }
        else{
            weights.emplace_back(- static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/float(maxWeight))) );
        }
    }

    bias = rand()% maxBias;
    if(rand()%2)
        bias *= -1;
    //ctor
}

Neuron::~Neuron  ()
{
    //dtor
}

float Neuron::sigmoid(float x) {
    return( 1/(1+float(exp(-x))));
}

float Neuron::sigmoidPrime(float x) {
    return( sigmoid(x)*(1-sigmoid(x)));
}



void Neuron::processActivations(const std::vector<std::vector<float>> &previousLayerActivations) {

    for(const std::vector<float> & layerActivation : previousLayerActivations){
        float sum=0;
        for(auto preActivation : hadamardProduct(layerActivation,weights)){
            sum += preActivation;

        }


        sum += bias;

        preActivation.emplace_back(sum);
        activations.emplace_back(sigmoid(sum)) ;
    }
}




std::vector<float> Neuron::hadamardProduct(const std::vector<float> & vector1, const std::vector<float> & vector2) {
    std::vector<float> result;
    try {
        if ( vector1.size() != vector2.size()){
            std::cout <<vector1.size() << " les vecteurs ne sont pas de la même taille "<<vector2.size() <<std::endl ;
            throw "les vecteurs ne sont pas de la même taille";}
    }
    catch(const std::string & message) {
        std::cout << message;
    }
    for(unsigned i =0 ; i<vector1.size() ; ++i )
        result.emplace_back(vector1[i] * vector2[i])  ;
    return result;

}


const std::vector<float> & Neuron::getActivations() const {
    return activations;
}




const float& Neuron::getActivation(const unsigned short adress) {
    return activations[adress];
}

std::vector<float> Neuron::getErrors() {
    return errors;
}

std::ostream &operator<<(std::ostream &os, const Neuron &neuron) {
    os << "neuron :";// << neuron.bias;
    //os<< neuron.error.size();
    for (auto data : neuron.activations ){

        os << data;
        os << "  ";
    }

    return os;

}

void Neuron::resetActivations() {
    activations.clear();
    errors.clear();
    preActivation.clear();
}

void Neuron::processLastNeuronError(const std::vector<float> &outputError, const bool &trainGenerator) {

    if(trainGenerator){
        for ( unsigned short i = 0; i < outputError.size() ; ++i ) {
            this->errors.emplace_back(this->getActivation(i) - outputError[i]);
        }
    }else{
        for ( unsigned short  i = 0; i < outputError.size() ; ++i ) {
            float error = outputError[i] - this->getActivation(i);
            if( error <0){
                this->errors.emplace_back(error +1  );
            }else{
                this->errors.emplace_back(error -1  );
            }

        }
    }

}
/**
 * process the changes on the weights and biaises
 * - learningRate * (1/numberOfFeedForwards) * (partials derivatives of the costs with respect to this neuron summed up)
 */
void Neuron::gradientDescent(const std::vector<std::vector<float>> &previousLayerActivations, const double &regularizationTerm,
                             const bool &trainingGenerator) {

    //bias update
    float meanError = 0;
    //faire la moyenne de erreur du neurone
    for(auto feedforwardError : errors){

        meanError += feedforwardError;
    }
    meanError  /= errors.size();

    bias += - Network::learningRate * meanError;

    //weights update
    //  there as much weights in the current neuron as the number of neurons in the previous layer
    //for every weights in this neuron
    for(unsigned weightNumber = 0 ; weightNumber< weights.size(); ++weightNumber){
        float weightChangesSummed = 0;
        //for every feedforward
        for(unsigned feedforwarNumber =0; feedforwarNumber< previousLayerActivations.size(); ++feedforwarNumber){
            weightChangesSummed += errors[feedforwarNumber] * previousLayerActivations[feedforwarNumber][weightNumber];
            //error for this feedforward * activation pour le neuron associé à ce poids dans le layer d'avant pour ce feedforward
        }
        //the value objective choose if we are lowering  or trying to increase the cost
        //we take the negative gradient if we want to decrease then improving the discriminator


        weights[weightNumber]  +=  -( (Network::learningRate/(errors.size())) *  weightChangesSummed
                -((Network::learningRate*regularizationTerm)/errors.size())*  weights[weightNumber]);

            //calcul --> -or+ (learningRate/feedforwards) *  (weightChangesSummed)-(learningRate * lamba)/feedforwards)*weight
    }

}
void Neuron::addActivation(const float &newActivation) {
    activations.emplace_back(newActivation);
}


std::vector<float> Neuron::getWeights() {
    return weights;
}

const std::vector<float> &Neuron::getPreActivation() const {
    return preActivation;
}

void Neuron::addError(float _error) {
    errors.emplace_back(_error);
}

void Neuron::debugSetBias(int newBias) {
    bias = newBias;
}

const std::vector<float> &Neuron::getErrors() const {
    return errors;
}
