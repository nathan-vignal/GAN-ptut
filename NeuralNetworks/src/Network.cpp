#include <iostream>
#include "Network.h"
#include <math.h>
using namespace std;

const float Network::learningRate = 0.15;
/**
 *
 * @param hiddenLayersArchitecture [[nombre de neuron][nombre neuron]] create a network of 2 layers without counting the input layer
 * @param _entries input data
 * @param _output data
 * @param _numberOfEpochs
 * @param _regularizationTerm this term is used to change the cost, it helps redure overfitting
 * @param _indexStartDiscriminator index+1 at which the discriminator begin
 */
Network::Network(const std::vector<unsigned short> &hiddenLayersArchitecture,
                 const std::vector<std::vector<float> > &_entries,
                 const std::vector<std::vector<float> > &_output, const unsigned int &_numberOfEpochs,
                 const double &_regularizationTerm, const unsigned &_indexStartDiscriminator) {

    indexStartDiscriminator = _indexStartDiscriminator;
    regularizationTerm = _regularizationTerm;
    numberOfEpochs = _numberOfEpochs;
    //distribution of the entry data on epochs
    for(unsigned i = 1 ; i<=numberOfEpochs; ++i){
        entries.emplace_back( std::vector<std::vector<float> > (&_entries[(_entries.size()/numberOfEpochs)*(i-1)],&_entries[(_entries.size()/numberOfEpochs)*(i)]));
    }
    //distribution of the output data on epochs
    for(unsigned i = 1 ; i<=numberOfEpochs; ++i){
        output.emplace_back( std::vector<std::vector<float> > (&_output[(_output.size()/numberOfEpochs)*(i-1)],&_output[(_output.size()/numberOfEpochs)*(i)]));
    }


    //separating real entries from fake ones
    for(unsigned numberOfTheEpoch=0; numberOfTheEpoch<output.size();++numberOfTheEpoch){
        std::vector<std::vector<float>> realEntriesEpoch;
        std::vector<std::vector<float>> fakeEntriesEpoch;

        for(unsigned feedForwardNumber =0;
            feedForwardNumber< entries[numberOfTheEpoch].size();
            ++feedForwardNumber){

            if(output[numberOfTheEpoch][feedForwardNumber][0] > 0.9){  // if the feedForward is real data
                realEntriesEpoch.emplace_back(entries[numberOfTheEpoch][feedForwardNumber]);
            }else{
                fakeEntriesEpoch.emplace_back(entries[numberOfTheEpoch][feedForwardNumber]);
            }
        }
        realEntries.emplace_back(realEntriesEpoch);
        fakeEntries.emplace_back(fakeEntriesEpoch);

    }

/*
    for(auto & first : entries){
        cout << "\n";
        for(auto & second : first){
            cout << "[ ";
            for( auto & data : second){
                cout << data <<" ";
            }
            cout << "]";
        }
    }

    std::cout <<'\n'<<'\n';
    for(auto & first : output){
        cout << "\n";
        for(auto & second : first){
            cout << "[ ";
            for( auto & data : second){
                cout << data <<" ";
            }
            cout << "]";
        }
    }
*/

    Layer *  firstLayer = new Layer(hiddenLayersArchitecture[0],(unsigned short)entries[0][0].size());
    layers.emplace_back(firstLayer );  //emplace_back plus opti que push_back

    for (unsigned i=1; i<hiddenLayersArchitecture.size(); ++i){
        auto * layer = new Layer(hiddenLayersArchitecture[i],hiddenLayersArchitecture[i-1]);
        layers.emplace_back( layer );  //emplace_back plus opti que push_back
    }
    Layer * lastLayer = new Layer((unsigned short)output[0][0].size(),hiddenLayersArchitecture[hiddenLayersArchitecture.size()-1]);
    layers.emplace_back(lastLayer );  //emplace_back plus opti que push_back
}//constructor


Network::~Network() {
    //dtor
}



void Network::feedforward(const unsigned int &numberOfTheEpoch, const bool &trainGenerator) {

    //going throught the generator and stopping at the begginning of the discriminator
    layers[0]->processMyNeuronsActivations(fakeEntries[numberOfTheEpoch]);
    for(unsigned y=1; y < indexStartDiscriminator; ++y){
        layers[y]->processMyNeuronsActivations(layers[y-1]->getMyactivations());
    }
    //inserting real data into the last generator layer as activation
    if(!trainGenerator){
        layers[indexStartDiscriminator-1]->addActivations(realEntries[numberOfTheEpoch]); // utile pour la backpropagation
    }
    //going throught the rest of the network(the disciminator)
    for(unsigned y=indexStartDiscriminator; y < layers.size(); ++y){
        layers[y]->processMyNeuronsActivations(layers[y-1]->getMyactivations());
    }



}//feedforward

/**
 * return the output of the generator or the discriminator for the given entries
 * @param entries test data to give to the generator
 * @param testGenerator if true we will test the generator, if false the discriminator
 * @return output of the generator
 */
vector<float> Network::testFeedforward(const std::vector<float> &entries, const bool &testGenerator) {

    if(testGenerator){ // we are testing the generator
        //process the first layer activation
        layers[0]->processMyNeuronsActivations({entries});
        //process the rest of the generator layers's activations
        for(unsigned i=1; i <= indexStartDiscriminator; ++i){
            layers[i]->processMyNeuronsActivations(layers[i-1]->getMyactivations());
        }
        std::vector<float>result = layers[indexStartDiscriminator-1]->getMyactivations()[0];
        resetActivations();
        return result;
    }else{//we are testing the discriminator

        //process the first layer for the generator
        layers[indexStartDiscriminator-1]->addActivations({entries});
        //process the rest of the network's activations
        for(unsigned i=indexStartDiscriminator; i < layers.size(); ++i){
            layers[i]->processMyNeuronsActivations(layers[i-1]->getMyactivations());
        }
         std::vector<float>result = layers[layers.size()-1]->getMyactivations()[0];
                 resetActivations();
        return result;
    }

}//testFeedforward


std::ostream& operator<< (std::ostream& stream, Network & network) {
    stream << '\n';
    stream << "\nNETWORK of "<< network.layers.size()<<" layers :" << '\n';
    for (const auto layer : network.layers) {
        stream << " \nLAYER ";
        stream <<  * layer<<endl;


    }
    stream << '\n';
    return stream;
}//surcharge <<

const vector<Layer *> &Network::getLayers() const {
    return layers;
}

/**
 * process network global cost for each feedforward in the given batchNumber
 */
void Network::processCost(const unsigned int &batchNumber) {
    costs.clear();
    std::vector<std::vector<float>> lastLayerActivations = getLayers()[getLayers().size()-1]->getMyactivations();
    //for each neuron in the last layer
    for(unsigned neuronNumber = 0; neuronNumber < lastLayerActivations[0].size(); ++neuronNumber ){
        float cost = 0;
        //for each feedForward
        for(unsigned feedForwardNumber = 0; feedForwardNumber < lastLayerActivations.size(); ++feedForwardNumber){//feedForwardNumber is the number of the entry that was used to process this activation
            if(lastLayerActivations[feedForwardNumber][neuronNumber] != 0 &&
            lastLayerActivations[feedForwardNumber][neuronNumber] != 1){// log(0) doesn't exist
                cost += (output[batchNumber][feedForwardNumber][neuronNumber] * log(lastLayerActivations[feedForwardNumber][neuronNumber])
                         + (1- output[batchNumber][feedForwardNumber][neuronNumber])* log(1- lastLayerActivations[feedForwardNumber][neuronNumber]));
                //  the formula used is cross entropy c = y * ln(a) + (1-y) * ln(1-a)
                //        y = output[feedForwardNumber][neuronNumber]  a= lastLayerActivations[feedForwardNumber][neuronNumber]
            }

        }
        //
        cost *= float(-1)/(lastLayerActivations.size());

        this->costs.emplace_back(cost);
    }

}
long double Network::processMeanError() {
    float meanError = 0;
    for(auto neuron : layers[layers.size()-1]->getNeurons()){
        for( auto neuronError : neuron->getErrors())
        {
            if(neuronError>0){
                meanError += neuronError;
            }else{
                meanError -= neuronError;
            }

        }
    }
    meanError /= layers[layers.size()-1]->getNeurons().size(); // meanError /= number of neurons
    meanError /= layers[layers.size()-1]->getNeurons()[0]->getActivations().size();// meanError /= number of feedforwards;
    return meanError;



}



const vector<float> &Network::getCost() const {
    return costs;
}
/*
void Network::vectorResizing(std::vector<std::vector<float>> vector, unsigned rows, unsigned columns) {
    vector.resize(rows);
    for ( auto &it : vector) {
        it.resize(columns);
    }
}*/


void Network::resetActivations() {
    for(Layer * layer : layers){
        layer->resetActivations();
    }
}

void Network::main() {

    bool trainGenerator = false;
    for(unsigned int numberOfTheEpoch=0 ; numberOfTheEpoch<numberOfEpochs; ++numberOfTheEpoch){


        feedforward(numberOfTheEpoch, trainGenerator);

        backPropagation(numberOfTheEpoch, trainGenerator);

        gradientDescent(numberOfTheEpoch, trainGenerator);

        if(numberOfTheEpoch > numberOfEpochs ){
            std::cout << *this;
        }
        long double meanError = processMeanError();
        if(numberOfTheEpoch%100 == 0){

            std::cout << '\n' << " numberOfTheEpoch " << numberOfTheEpoch << " mean error " << meanError
            << endl;
            trainGenerator ? std::cout<<"Trainning the Generator" : std::cout<<"  Training the Discriminator"; std::cout <<"\n";
        }
        //statement can be simplified but it's easier to read that way ;)
        if(meanError<0.2){ //if the current network to train is doing well
            trainGenerator = !trainGenerator; //we change to network to train
            cout <<" changing sides" ;
        }
        resetActivations();
    }
}

void Network::backPropagation(const unsigned int &numberOfTheEpoch, const bool &trainGenerator) {
    //process the partial derivative with respect to z for each layer

    //if we want to train the generator we can throw the real data
    if(trainGenerator){
        std::vector<std::vector<float>> outputToTrainOn ;
        for(auto data : output[numberOfTheEpoch]){
            if(data[0] <0.1)
                outputToTrainOn.emplace_back(data);
        }
        layers[layers.size() - 1]->processLastLayerError(outputToTrainOn,true); //process the partial derivative of c with respect to z for the last layer
        //std::cout << * layers[layers.size()-1];
        for(unsigned i = (unsigned)layers.size()-2 ; i >= 0 && i<999999; --i){ //use the partial derivative c/z of the n+1 layer to process it for n
            //condition i<99999 car 0-1 >0 dans le référentiel des unsigneds
            layers[i]->processLayerError(  layers[i+1] );
        }
    }else{ //if we are training the discriminator we don't backpropagate all the way to the beginning

        layers[layers.size() - 1]->processLastLayerError(output[numberOfTheEpoch], false); //process the partial derivative of c with respect to z for the last layer

        //std::cout << * layers[layers.size()-1];
        for(int i = layers.size()-2 ; i >= indexStartDiscriminator  ; --i){ //use the partial derivative c/z of the n+1 layer to process it for n

            layers[i]->processLayerError(  layers[i+1] );
        }

    }



}
/**
 *
 * @param batchNumber number of the batch we are trying to apply gradient descent to
 * @param trainGenerator  true if we are applying gradient descent to train the generator,
 * false if we are training the discriminator
 */
void Network::gradientDescent(const unsigned int &batchNumber, const bool &trainGenerator) {

    if(trainGenerator){//if we are training the generator
        for(unsigned layerNumber =indexStartDiscriminator-1; layerNumber>0 ;--layerNumber  ){
            layers[layerNumber]->layerGradientDescent(layers[layerNumber - 1]->getMyactivations(), regularizationTerm,
                                                      trainGenerator);
        }
        layers[0]->layerGradientDescent(entries[batchNumber], regularizationTerm, trainGenerator);
    }else{//if we are training the discriminator


        for(unsigned layerNumber =layers.size()-1; layerNumber>=indexStartDiscriminator;--layerNumber  ){
            layers[layerNumber]->layerGradientDescent(layers[layerNumber - 1]->getMyactivations(), regularizationTerm,
                                                      trainGenerator);
        }

    }



}








