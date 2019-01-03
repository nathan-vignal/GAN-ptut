#include <iostream>
#include <vector>
#include "Network.h"

using namespace std;

int main()
{

    vector<vector<float>> input = {{0,0.8,0.3,0.9},{1,0.8,0.6,0.4},{0.1,0.2,0.2,0.6},{0.1,0.2,0.2,0.6}};
    vector<vector<float>> output = {{0,1,1,0},{1,0,1,0},{1,1,1,0},{0,0,1,0}};
                    //Network(nombre de layer, nombre de neuron par layer, input, output)
    Network network((unsigned short)5,(unsigned short)10,input,output,2);
    network.main();
    cout << network;
    for(auto neuronCost : network.getCost())
        cout << neuronCost<< ' ';


    return 0;
}
