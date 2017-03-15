#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <math.h>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <random>
#include <omp.h>

static const int BIG = 10000;
static const int SMALL = -10000;
static const int nJOINT = 31;
static const double THREASHOLD = 0.05;

using namespace std;

double poseDist(vector<double>, vector<double>);
double refreshMinDists(vector<double>, vector<double>, double);
vector<double> scalingWC(vector<double>, double);
void inputWCData(string, vector< vector<double> >&, vector< pair<string, int> >&, string);
vector<double> split(string, const char);
void copyHierarchy(string, string);
vector<int> FNClustering(vector< vector<double> >);


double poseDist(vector<double> data1, vector<double> data2)
{
    double maxD = 0.0;
    double data_dif;
    for(int i=0; i<nJOINT; i++){
        double d = 0;
        for(int j=0; j<3; j++){
            data_dif = data1[3*i+j] - data2[3*i+j];
            d += data_dif * data_dif;
        }
        d = sqrt(d);
        maxD = max(d, maxD);
    }
    return maxD;
}

double refreshMinDists(vector<double> data1, vector<double> data2, double minDist)
{
    return min(poseDist(data1, data2), minDist);
}


vector<double> scalingWC(vector<double> posList, double height)
{
    for(int i = 0; i < posList.size(); i++)
        posList[i] = posList[i] * 1.7 / height;
    return posList;
}


vector<double> split(string s, const char delim)
{
    vector<double> elems;
    stringstream ss(s);
    string item;
    double dItem;

    while(getline(ss, item, delim)){
        if(!item.empty()){
            stringstream ssItem(item);
            ssItem >> dItem;
            elems.push_back(dItem);
        }
    }

    return elems;
}


void inputWCData(string WCFile, vector< vector<double> > &data, vector< pair<string, int> > &indices, string filename)
{

    int i = 0;
    double height;
    vector<double> WCParms;
    ifstream ifs(WCFile);
    string line;
    pair<string, int> index(filename, 0);
    
    while(getline(ifs, line)){
        WCParms = split(line, ' ');
        if(i == 0){
            double top = SMALL;
            double bottom = BIG;
            
            for(int j = 1; j < WCParms.size(); j += 3){
                top = max(top, WCParms[j]);
                bottom = min(bottom, WCParms[j]);
            }
            height = top - bottom;
        }
        WCParms = scalingWC(WCParms, height);
        data.push_back(WCParms);
        index.second = i;
        indices.push_back(index);
        i++;
    }

}


void copyHierarchy(string inFile, string outFile)
{
    ifstream ifs(inFile);
    ofstream ofs(outFile);
    string line;

    while(getline(ifs, line)){
       if(line.find("MOTION") != string::npos) 
           break;
       else
           ofs << line << endl;
    }
}


vector<int> FNClustering(vector< vector<double> > data)
{
    random_device rnd;
    mt19937 mt(rnd());
    int nData = data.size();
    uniform_real_distribution<> rand_init(0, nData);
    int argMaxDist = rand_init(mt);
    vector<int> outIndex;
    outIndex.push_back(argMaxDist);
    vector<int> restIndex;
    for(int i = 0; i < nData; i++)
        if(i != argMaxDist)
            restIndex.push_back(i);
    vector<double> minDists(nData, BIG);
    minDists[argMaxDist] = 0;
    double maxDist = BIG;

    while(maxDist > THREASHOLD){
        clock_t iterStart = clock();
        #pragma omp parallel for
        for(int i = 0; i < restIndex.size(); i++)
            minDists[restIndex[i]] = refreshMinDists(data[restIndex[i]], data[argMaxDist], minDists[restIndex[i]]);
        cout << "Parallel calculation time: " << (double)(clock() - iterStart)/CLOCKS_PER_SEC << endl;

        maxDist = 0;
        for(int i = 0; i < restIndex.size(); i++){
            if(maxDist < minDists[restIndex[i]]){
                maxDist = minDists[restIndex[i]];
                argMaxDist = i;
            }
        }

        cout << "Add pose " << restIndex[argMaxDist] << endl;
        cout << "maxDist = " << maxDist << endl;
        outIndex.push_back(restIndex[argMaxDist]);
        restIndex.erase(restIndex.begin() + argMaxDist);
        minDists[restIndex[argMaxDist]] = 0;
        cout << "Single iteration time: " << (double)(clock() - iterStart)/CLOCKS_PER_SEC << endl;
    }
    cout << "Clustering " << nData << " -> " << outIndex.size() << endl;

    return outIndex;
}

int main()
{
    string IMpath = "../Data/Intermediate/";
    string RGpath = "../Data/Regularized/";
    string inAppendix = "_reduce.bvh";
    string outAppendix = ".bvh";
    string WCAppendix = "_WC";

    cout << "Data input..." << endl;
    string line;
    clock_t inStart = clock();
    vector<string> fNames;

    ifstream ifs("./filenames.dat");
    while(getline(ifs, line)){
        fNames.push_back(line); 
    }

    vector< vector<double> > data;
    vector< pair<string, int> > indices;

    for(int i = 0; i < fNames.size(); i++){
    
        string inFile = IMpath + fNames[i] + inAppendix;
        string outFile = RGpath + fNames[i] + outAppendix;
        copyHierarchy(inFile, outFile);

        string WCFile = IMpath + fNames[i] + WCAppendix;

        cout << fNames[i] << endl;
        inputWCData(WCFile, data, indices, fNames[i]);
    }

    cout << "Input " << data.size() << " motion data." << endl;
    cout << "Input time: " << (double)(clock() - inStart)/CLOCKS_PER_SEC << endl;

    cout << "Clustering..." << endl;
    clock_t clStart = clock();
    
    vector<int> outIndex = FNClustering(data);

    cout << "Clustering time: " << (double)(clock() - clStart)/CLOCKS_PER_SEC << endl;

    cout << "Saving..." << endl;
    map<string, int> mCount;

    for(int i = 0; i < fNames.size(); i++)
        mCount[fNames[i]] = 0;

    for(int i = 0; i < outIndex.size(); i++){
        pair<string, int> index = indices[outIndex[i]];
        mCount[index.first]++;
    }

    for(int i = 0; i < fNames.size(); i++){
        ofstream ofs(RGpath + fNames[i] + ".bvh", ios::app);
        ofs << "MOTION" << endl;
        ofs << "Frames: " << mCount[fNames[i]] << endl;
        ofs << "Frame Time: 0.0333332" << endl;
        ifstream ifs(IMpath + fNames[i] + inAppendix);
        vector<string> lines;
        bool motionFlag = false;
        while(getline(ifs, line)){
            if(motionFlag)
                lines.push_back(line);
            else if(line.find("Frame Time:", 0) != string::npos)
                motionFlag = true;
        }

        for(int j = 0; j < outIndex.size(); j++){
            pair<string, int> index = indices[outIndex[j]];
            cout << fNames[i] << ", " << index.first << endl;
            if(fNames[i].compare(index.first)==0){
                ofs << lines[index.second] << endl;
            }
        }
    }

    ofstream indexOfs("./outIndices.dat");
    for(int i = 0; i < outIndex.size(); i++){
        pair<string, int> index = indices[outIndex[i]];
        ofstream ofs(RGpath+index.first+".bvh", ios::app);
        indexOfs << "(" << index.first << ", " << index.second << ")" << endl;
    }
    cout << "Done!!!" << endl;
    cout << "Total time: " << (double)(clock() - inStart)/CLOCKS_PER_SEC << endl;
}

