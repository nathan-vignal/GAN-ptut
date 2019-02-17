#include <iostream>
#include <vector>
#include <ctime>
#include "Network.h"
#include "Feeder.h"
using namespace std;

int main()
{
    srand (static_cast <unsigned> (time(0))); //to create a more real randomness
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
    Network network({20,20,2,20,20}, input, output, 10000, 0.001, 3);
    //network.feedforward(0)
    //network.backPropagation(0)

    network.main();
    std::cout<< '\n';
    std::cout<< '\n';
    //variable used to block the loop below with the cin and free it

    std::vector<std::vector<float>>  fakeInputs;
    for(unsigned i =0; i<50; ++i ){
        std::vector<float> temp = {static_cast <float> (rand()) / static_cast <float> (RAND_MAX),static_cast <float> (rand()) / static_cast <float> (RAND_MAX)};
        fakeInputs.emplace_back(temp) ;//{0,0};

    }

    std::vector<std::vector<float>>  realInputs;
    for(unsigned i =0; i<50; ++i ){
        float  x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        std::vector<float> temp ={x,x*x}; //{1,1};//
        realInputs.emplace_back(temp) ;
    }

    float sumResult = 0;
    for(const auto  & fakeInput: fakeInputs){
        sumResult+= network.testFeedforward(network.testFeedforward(fakeInput,true),false)[0];
    }
    cout << "mean result for fake inputs :"<< sumResult/ (fakeInputs.size()+1)<<endl;

    sumResult = 0;
    for(const auto  & realInput: realInputs){

        sumResult+= network.testFeedforward(realInput,false)[0];
    }
    cout << "mean result for real inputs :"<< (sumResult/(realInputs.size()+1)) <<endl ;
    cout << "\n \n ";
    //test génération
    for(unsigned i = 0 ; i<3 ; ++i) {
        cout << "\n";
        for (auto data : network.testFeedforward({static_cast <float> (rand()) / static_cast <float> (RAND_MAX),
                                                  static_cast <float> (rand()) / static_cast <float> (RAND_MAX)},
                                                 true)) {
            std::cout << data <<" ";
        };
    }
        return 0;
    }
