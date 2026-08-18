#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <set>
#include <unordered_map>
#include <map>
#include <omp.h>
using namespace std;
const int TARGET_VOCAB = 16000;
bool isBoundary(const string &x){
    static const set<string> b = {
        "。","，","、","；","：","？","！","…","—",
        "“","”","‘","’","（","）","《","》",
        "\n","\r"," ","　",
        ".",",",";",":","?","!"
    };
    return b.count(x) > 0;
}
void split(const string &x, vector<vector<string>> &corpus){
    int len = 0;
    vector<string> cur;
    for(size_t i = 0;i < x.size();i+=len){
        unsigned char t = x[i];
        if(t < 128) len = 1;
        else if(t < 224) len = 2;
        else if(t < 240) len = 3;
        else len = 4;

        string byte = x.substr(i, len);

        if(!isBoundary(byte)){
            cur.push_back(byte);
        }else{
            if(!cur.empty()){
                corpus.push_back(cur);
                cur.clear();
            }
            if(byte != " " && byte != "\n" && byte != "\r" && byte != "　"){
                corpus.push_back({byte});
            }
        }
    }
    if(!cur.empty()){
        corpus.push_back(cur);
    }
}
void special_tokens(vector<string> &id2token, unordered_map<string, int> &token2id){
    vector<string> s = {"<|unk|>", "<|user|>", "<|assistant|>", "<|end|>"};
    for(int i = 0;i < s.size();i++){
        token2id[s[i]] = id2token.size();
        id2token.push_back(s[i]);
    }
}
void char2id(vector<vector<string>> &corpusOrigin, vector<vector<int>> &corpus, vector<string> &id2token, unordered_map<string, int> &token2id){
    for(int i = 0;i < corpusOrigin.size();i++){
        corpus.push_back({});
        for(int j = 0;j < corpusOrigin[i].size();j++){
            const string &ch = corpusOrigin[i][j];
            auto it = token2id.find(ch);
            if(it == token2id.end()){
                token2id[ch] = id2token.size();
                corpus[i].push_back(id2token.size());
                id2token.push_back(ch);
            }else{
                corpus[i].push_back(it -> second);
            }
        }
    }
}
map<pair<int, int>, int> countPairs(const vector<vector<int>> &corpus){
    map<pair<int, int>, int> freq;
    for(int i = 0;i < corpus.size();i++){
        for(int j = 0;j + 1 < corpus[i].size();j++){
            int a = corpus[i][j];
            int b = corpus[i][j+1];
            freq[{a, b}]++;
        }
    }
    return freq;
}
void mergePair(vector<vector<int>> &corpus, int a, int b, int newId){
    for(int i = 0;i < corpus.size();i++){
        vector<int> out;
        for(int j = 0;j < corpus[i].size();){
            if(j + 1 < corpus[i].size() && corpus[i][j] == a && corpus[i][j+1] == b){
                out.push_back(newId);
                j += 2;
            }else{
                out.push_back(corpus[i][j]);
                j += 1;
            }
        }
        corpus[i] = out;
    }
}
void saveVocab(const string &path, const vector<string> &id2token){
    ofstream f(path);
    f << id2token.size() << endl;
    for(int i = 0;i < id2token.size();i++)
        f << i << ' ' << id2token[i] << endl;
    f.close();
}

void saveMerges(const string &path, const vector<pair<pair<int,int>,int>> &merges){
    ofstream f(path);
    f << merges.size() << endl;
    for(int i = 0;i < merges.size();i++)
        f << merges[i].first.first << ' '
          << merges[i].first.second << ' '
          << merges[i].second << endl;
    f.close();
}
int main()
{
    string textPath = "train/train.txt";
    ifstream textOrigin(textPath);
    stringstream buffer;
    buffer << textOrigin.rdbuf();
    string text = buffer.str();

    vector<vector<string>> corpusOrigin;
    vector<vector<int>> corpus;
    vector<string> id2token;
    unordered_map<string, int> token2id;

    special_tokens(id2token, token2id);
    split(text, corpusOrigin);
    char2id(corpusOrigin, corpus, id2token, token2id);

    cout << "vocab: " << id2token.size() << endl;
    cout << "segments: " << corpus.size() << endl;

    long long total = 0;
    for(auto &s : corpus) total += s.size();
    cout << "total tokens: " << total << endl;

    vector<pair<pair<int,int>, int>> merges;
    while(id2token.size() < TARGET_VOCAB){
        map<pair<int,int>, int> freq = countPairs(corpus);
        if(freq.empty()) break;

        int bestA = -1, bestB = -1, bestCount = 0;
        for(map<pair<int, int>, int>::iterator it = freq.begin(); it != freq.end();it++){
            if(it -> second > bestCount){
                bestCount = it -> second;
                bestA = it -> first.first;
                bestB = it -> first.second;
            }
        }
        if(bestCount < 5) break;

        int newId = id2token.size();
        id2token.push_back(id2token[bestA] + id2token[bestB]);
        merges.push_back({{bestA, bestB}, newId});
        mergePair(corpus, bestA, bestB, newId);

        cout <<merges.size() << " " << id2token[newId] << " freq=" << bestCount <<endl;
    }

    saveVocab("train/bpe_vocab.txt", id2token);
    saveMerges("train/bpe_merges.txt", merges);

    long long after = 0;
    for(int i = 0;i < corpus.size();i++) after += corpus[i].size();
    cout << "final tokens: " << after << endl;
    cout << "compression: " << (double)total / after << endl;

    return 0;
}