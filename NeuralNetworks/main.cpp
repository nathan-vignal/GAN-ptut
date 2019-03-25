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




//Network(architecture, input data, expected input, nb batch, regularization term, indexStart Discriminator)
    Network network({20,2,20}, input, output, 10000, 0.1, 2);

    //regularization term

    network.main();
    std::cout<< '\n';
    std::cout<< '\n';

    //here begin the testing of our final network
    //creating fake inputs
    std::vector<std::vector<float>>  fakeInputs;
    for(unsigned i =0; i<50; ++i ){
        std::vector<float> temp = {static_cast <float> (rand()) / static_cast <float> (RAND_MAX),static_cast <float> (rand()) / static_cast <float> (RAND_MAX)};
        fakeInputs.emplace_back(temp) ;//{0,0};

    }
    //creating realInputs
    std::vector<std::vector<float>>  realInputs;
    for(unsigned i =0; i<50; ++i ){
        float  x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        std::vector<float> temp ={x,x*x}; //{1,1};//
        realInputs.emplace_back(temp) ;
    }

    //testing the performance of the discriminator
    float sumResult = 0;
    for(const auto  & fakeInput: fakeInputs){
        sumResult+= network.testFeedforward(network.testFeedforward(fakeInput,true),false)[0];
    }
    cout << "mean result for fake inputs :"<< sumResult/ (fakeInputs.size())<<endl;

    sumResult = 0;
    for(const auto  & realInput: realInputs){

        sumResult+= network.testFeedforward(realInput,false)[0];
    }
    cout << "mean result for real inputs :"<< (sumResult/(realInputs.size())) <<endl ;
    cout << "\n \n ";
    //testing the generator
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
