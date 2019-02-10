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
                    //Network(nombre de layer, nombre de neuron par layer, input, output)

    Network network({2,2}, input, output, 5, 0.1, 2);
    //network.feedforward(0);
    //network.backPropagation(0)

    cout << network;
    network.main();
    std::cout<< '\n';
    while(1){
        float first = 0;
        float second = 0;
        float third = 0;
        float four = 0;

        cout << "veuillez entrer un vecteur ";
        cin >> first;
        cin >> second;
        cin >> third;
        cin >> four;
        for(auto neuronResult :network.testFeedforward({first,second,third,four}) ){
            cout<< '\n';
            for( auto data : neuronResult)
                cout<< " "<< data  ;
        }



    }
    cout<< "1110";

    cout<< "\n\n1000";
    for(auto neuronResult :network.testFeedforward({1,0,0,0}) ){
        cout<< '\n';
        for( auto data : neuronResult)
            cout<< " "<< data  ;
    }
    cout<< "0110";
    for(auto neuronResult :network.testFeedforward({0,1,1,0}) ){
        cout<< '\n';
        for( auto data : neuronResult)
            cout<< " "<< data  ;
    }



    return 0;
}
