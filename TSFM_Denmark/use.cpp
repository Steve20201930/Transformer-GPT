#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <algorithm>
#include <random>
using namespace std;

typedef vector<vector<float>> Matrix;
int vocab_size = 0, d = 64, dk = 64;
const int h = 1; //The Multi-head quantity must be checked
const int max_seq_len = 120;
const int epoch = 10000;
const int record = 500;

Matrix matmul(const Matrix &A, const Matrix &B){
    int Arow = A.size();
    int Acolumn = A[0].size();
    int Brow = B.size();
    int Bcolumn = B[0].size();
    Matrix res(Arow, vector<float>(Bcolumn, 0.0f));

    for(int ai = 0;ai < Arow;ai++){
        for(int bj = 0;bj < Bcolumn;bj++){
            float plus = 0.0f;
            for(int ajbi = 0;ajbi < Acolumn;ajbi++){
                plus += A[ai][ajbi] * B[ajbi][bj];
            }
            res[ai][bj] = plus;
        }
    }

    return res;
}
Matrix transpose(const Matrix &A){
    int Arow = A.size();
    int Acolumn = A[0].size();
    Matrix res(Acolumn, vector<float>(Arow, 0.0f));

    for(int i = 0;i < Arow; i++){
        for(int j = 0;j < Acolumn;j++){
            res[j][i] = A[i][j];
        }
    }

    return res;
}
Matrix softmax(const Matrix &A){
    int Arow = A.size();
    int Acolumn = A[0].size();
    Matrix res(Arow, vector<float>(Acolumn, 0.0f));

    for(int i = 0;i < Arow;i++){
        float e_sum = 0.0f;
        float maxv = A[i][0];
        for(int j = 0;j < Acolumn;j++){
            maxv = max(A[i][j], maxv);
        }
        for(int j = 0;j < Acolumn;j++){
            e_sum += exp(A[i][j] - maxv);
        }
        for(int j = 0;j < Acolumn;j++){
            res[i][j] = exp(A[i][j] - maxv) / e_sum;
        }
    }

    return res;
}
Matrix applyCausalMask(const Matrix &CM){
    int n = CM.size(); //Width and Height are the same size
    Matrix res(n, vector<float>(n, -1e9f));

    for(int i = 0;i < n;i++){
        for(int j = 0;j <= i;j++){
            res[i][j] = CM[i][j];
        }
    }
    return res;
}
Matrix attention(const Matrix &X, const Matrix &WQ, const Matrix &WK, const Matrix &WV, Matrix &Q, Matrix &K, Matrix &Kt, Matrix &V, Matrix &scores){
    /*
        Attention = Softmax((Q x Kt)/sqrt(dk)) x V
    */
    int n = X.size();
    int d = X[0].size();
    int dk = WQ[0].size();

    Q = matmul(X,WQ); // X x WQ
    K = matmul(X,WK); // X x WK
    V = matmul(X,WV); // X x WV
    Kt = transpose(K); // Kt

    scores = matmul(Q, Kt);
    
    //Divided by sqrt(dk) and softmax
    for(int i = 0;i < n;i++){
        for(int j = 0;j < n;j++){
            scores[i][j] /= sqrt(dk);
        }
    }
    scores = applyCausalMask(scores);
    scores = softmax(scores);

    Matrix res = matmul(scores, V);
    return res;
}

Matrix Matrix_load(string filename){
    ifstream file(filename);
    int a, b;
    file >> a >> b;

    Matrix res(a, vector<float>(b));
    for(int i = 0;i < a;i++){
        for(int j = 0;j < b;j++){
            file >> res[i][j];
        }
    }
    return res;
}
string Text_load(string filename){
    ifstream file(filename);
    stringstream buffer;
    buffer << file.rdbuf();
    string text = buffer.str();
    return text;
}
vector<string> uft8_split(const string &text){
    vector<string> res;
    int len = 0;
    for(size_t i = 0;i < text.size();i+=len){
        unsigned char t = text[i];
        if(t < 128) len = 1;
        else if(t < 224) len = 2;
        else if(t < 240) len = 3;
        else len = 4;

        string byte = text.substr(i, len);
        res.push_back(byte);
    }
    return res;
}
void Vocab_load(string filename, map<string, int> &char2id, vector<string> &id2char, int &vocab_size_link){
    ifstream file(filename);
    string str;

    getline(file, str);
    int size = stoi(str);

    for(int i = 0;i < size;i++){
        getline(file, str);
        id2char.push_back(str);
        char2id[str] = i;
    }
    file.close();

    vocab_size_link = id2char.size();
}
vector<int> tokenize(const string &text, map<string, int> &char2id){
    vector<string> original_tokens = uft8_split(text);
    vector<int> id_tokens;
    for(int i = 0;i < original_tokens.size();i++){
        if(char2id.count(original_tokens[i]) == 0){
            continue;
        }
        id_tokens.push_back(char2id[original_tokens[i]]);
    }

    return id_tokens;
}
Matrix embedLookup(const vector<int> &window, const Matrix &tokenTable){
    Matrix res(window.size(), vector<float>(tokenTable[0].size()));

    for(int i = 0;i < window.size();i++){
        res[i] = tokenTable[window[i]];
    }
    return res;
}
Matrix matrix_add(const Matrix &A, const Matrix &B){
    Matrix res(A.size(), vector<float>(A[0].size()));
    for(int i = 0;i < A.size();i++){
        for(int j = 0;j < A[0].size();j++){
            res[i][j] = A[i][j] + B[i][j];
        }
    }
    return res;
}
int find_next(const Matrix &result){
    int res = 0;
    float max = result[result.size()-1][0];
    for(int i = 1;i < result[0].size();i++){
        if(result[result.size()-1][i] > max){
            res = i;
            max = result[result.size()-1][i];
        }
    }
    return res;
}
int main()
{
    random_device rd;
    mt19937 gen(rd());
    /*Users zone*/
    string front = "我说：";
    string opposite = "，你说：";

    cout <<"> ";
    string usr;
    cin >> usr;
    string input = front + usr + opposite;

    /*      Loading data     */
    map<string, int> char2id;
    vector<string> id2char;
    Vocab_load("usr/Vocab.txt", char2id, id2char, vocab_size);
    Matrix tokenEmbedding = Matrix_load("usr/tokenEmbedding.txt");
    Matrix posEmbedding = Matrix_load("usr/posEmbedding.txt");
    Matrix WQ = Matrix_load("usr/WQ.txt");
    Matrix WK = Matrix_load("usr/WK.txt");
    Matrix WV = Matrix_load("usr/WV.txt");
    Matrix Wout = Matrix_load("usr/Wout.txt");

    string text = Text_load("usr/train.txt");
    vector<int> tokenIds = tokenize(text, char2id);

    vector<int> initialIds = tokenize(input, char2id);
    int remainingSlots = max_seq_len - initialIds.size();

    for(int m = 0;m < remainingSlots;m++){
        /*      Forward     */
        vector<int> tokenIds = tokenize(input, char2id);
        
        Matrix X = matrix_add(embedLookup(tokenIds, tokenEmbedding), posEmbedding);
        Matrix Q, K, Kt, V, scores;
        Matrix A = attention(X, WQ, WK, WV, Q, K, Kt, V, scores);
        Matrix result = matmul(A, Wout);

        int next = find_next(result);
        cout << id2char[next];
        cout.flush();
        input += id2char[next];
    }
    return 0;
}