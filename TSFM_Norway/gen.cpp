#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <map>
#include <sstream>
using namespace std;

typedef vector<vector<float>> Matrix;
int vocab_size = 0, d = 64, dk = 64;
int max_seq_len = 120;
int dimension = 4 * d;

void buildVocab(const string &text, int &vocab_size){
    /*utf8_split*/
    vector<string> split;
    int len = 0;
    for(size_t i = 0;i < text.size();i+=len){
        unsigned char t = text[i];
        if(t < 128) len = 1;
        else if(t < 224) len = 2;
        else if(t < 240) len = 3;
        else len = 4;

        string byte = text.substr(i, len);
        split.push_back(byte);
    }
    map<string, int> char2id;
    vector<string> id2char;
    for(size_t i = 0;i < split.size();i++){
        if(char2id.count(split[i]) == 0){
            int newId = char2id.size();
            char2id[split[i]] = newId;
            id2char.push_back(split[i]);
        }
    }

    ofstream file("train/Vocab.txt");
    file << char2id.size() <<endl;
    for(int i = 0;i < char2id.size();i++){
        file << id2char[i] <<endl;
    }
    file.close();

    vocab_size = char2id.size();
}
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
        Generating vocabulary library
    */
    buildVocab(read("train/train.txt"), vocab_size);
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
        Generating Wout/Wo file
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
    /*
        Generating WQ/K/V series files
    */
    ofstream WQ("train/WQ.txt");
    ofstream WK("train/WK.txt");
    ofstream WV("train/WV.txt");
    ofstream Wo("train/Wo.txt");
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
    ofstream gamma("train/gamma.txt");
    ofstream beta("train/beta.txt");
    ofstream gamma2("train/gamma2.txt");
    ofstream beta2("train/beta2.txt");
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
    ofstream Wup("train/Wup.txt");
    ofstream Wdown("train/Wdown.txt");
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
    return 0;
}