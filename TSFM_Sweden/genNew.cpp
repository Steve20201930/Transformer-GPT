#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <map>
#include <sstream>
using namespace std;

typedef vector<vector<float>> Matrix;
int vocab_size = 0, d = 384, dk = 384;
int max_seq_len = 256;
int dimension = 4 * d;
const int N = 8;
string read(string filename){
    ifstream file(filename);
    stringstream buffer;
    buffer << file.rdbuf();
    string text = buffer.str();
    return text;
}
int main()
{
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> dist(0.0f, 0.02f);
    /*
        Read real vocab size
    */
    ifstream vocabRead("train/bpe_vocab.txt");
    vocabRead >> vocab_size;
    vocabRead.close();
    /*
        Generating token embedding file
    */
    ofstream tokenEmbedding("train/tokenEmbedding.txt");
    tokenEmbedding << vocab_size << ' '<< d <<endl;
    for(int i = 0;i < vocab_size;i++){
        for(int j = 0;j < d;j++){
            tokenEmbedding << dist(gen) <<' ';
        }
        tokenEmbedding <<endl;
    }
    tokenEmbedding.close();
    /*
        Generating position embedding file
    */
    ofstream posEmbedding("train/posEmbedding.txt");
    posEmbedding << max_seq_len << ' '<< d <<endl;
    for(int i = 0;i < max_seq_len;i++){
        for(int j = 0;j < d;j++){
            posEmbedding << dist(gen) <<' ';
        }
        posEmbedding <<endl;
    }
    posEmbedding.close();
    /*
        Generating Wout file
    */

    ofstream Wout("train/Wout.txt");
    Wout << d <<' ' << vocab_size <<endl;
    for(int i = 0;i < d;i++){
        for(int j = 0;j < vocab_size;j++){
            Wout << dist(gen) <<' ';
        }
        Wout <<endl;
    }
    Wout.close();
    for(int l = 0;l < N;l++){
        /*
            Generating WQ/K/V/o series files
        */
        ofstream WQ("train/WQ_" + to_string(l) + ".txt");
        ofstream WK("train/WK_" + to_string(l) + ".txt");
        ofstream WV("train/WV_" + to_string(l) + ".txt");
        ofstream Wo("train/Wo_" + to_string(l) + ".txt");
        WQ << d <<' ' << dk <<endl;
        WK << d <<' ' << dk <<endl;
        WV << d <<' ' << dk <<endl;
        Wo << d <<' ' << dk <<endl;
        for(int i = 0;i < d;i++){
            for(int j = 0;j < dk;j++){
                WQ << dist(gen) << ' ';
                WK << dist(gen) << ' ';
                WV << dist(gen) << ' ';
                Wo << dist(gen) << ' ';
            }
            WQ <<endl;
            WK <<endl;
            WV <<endl;
            Wo <<endl;
        }
        WQ.close();
        WK.close();
        WV.close();
        Wo.close();
        /*
            Generating gamma/beta files
        */
        ofstream gamma("train/gamma_" + to_string(l) + ".txt");
        ofstream beta("train/beta_" + to_string(l) + ".txt");
        ofstream gamma2("train/gamma2_" + to_string(l) + ".txt");
        ofstream beta2("train/beta2_" + to_string(l) + ".txt");
        gamma << 1 << ' ' << d <<endl;
        beta << 1 << ' ' << d <<endl;
        gamma2 << 1 << ' ' << d <<endl;
        beta2 << 1 << ' ' << d <<endl;
        for(int i = 0;i < d;i++){
            gamma << 1.0 << ' ';
            beta << 0.0 << ' ';
            gamma2 << 1.0 << ' ';
            beta2 << 0.0 << ' ';
        }
        gamma.close();
        beta.close();
        gamma2.close();
        beta2.close();
        /*
            Generating Up/Down dimension files
        */
        ofstream Wup("train/Wup_" + to_string(l) + ".txt");
        ofstream Wdown("train/Wdown_" + to_string(l) + ".txt");
        Wup << d << ' ' << dimension <<endl;
        Wdown << dimension << ' ' << d <<endl;
        for(int i = 0;i < d;i++){
            for(int j = 0;j < dimension;j++){
                Wup << dist(gen) << ' ';
            }
            Wup << endl;
        }
        for(int i = 0;i < dimension;i++){
            for(int j = 0;j < d;j++){
                Wdown << dist(gen) << ' ';
            }
            Wdown <<endl;
        }
        Wup.close();
        Wdown.close();
    }
    return 0;
}