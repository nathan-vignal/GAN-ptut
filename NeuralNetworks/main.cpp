#include <iostream>
#include <vector>
#include "Network.h"
#include "Feeder.h"
using namespace std;

int main()
{
    vector<vector<float>> input ;
    vector<vector<float>> output ;
    Feeder::createData( input,  output);
    /*
    for(auto truc : input){
        cout << '\n';
        for(auto truc2: truc){
            cout << truc2 << " ";
        }
    }
     */



    /* vector<vector<float>> input = {{0,0.8,0.3,0.9},{1,0.8,0.6,0.4},{0.1,0.2,0.2,0.6},{0.1,0.2,0.2,0.6}};
     vector<vector<float>> output = {{0,1,1,0},{1,0,1,0},{1,1,1,0},{0,0,1,0}};*/

//Network(architecture, input data, expected input, nb batch, regularization term, indexStart Discriminator)
    Network network({2,2,4,4}, input, output, 7000, 0.0001, 2);
    //network.feedforward(0);
    //network.backPropagation(0)

    cout << network;
    network.main();
    std::cout<< '\n';
    std::cout<< '\n';
    //variable used to block the loop below with the cin and free it
    /*
    unsigned interupt ;
    while(1){


    float first = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    float second = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

    for(const auto & neuronResult :network.testFeedforward({first,second}) ){
        cout<< '\n';
        for( auto data : neuronResult)
            cout<< " "<< data  ;
    }
    cin >> interupt;

    std::cout<< "\n";
    }*/

    return 0;
}
