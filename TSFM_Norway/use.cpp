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
const int h = 8; //The Multi-head quantity must be checked
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
Matrix concat(const Matrix &A, const Matrix &B){
    Matrix res(A.size(), vector<float>(A[0].size()+B[0].size()));
    for(int i = 0;i < A.size();i++){
        for(int j = 0;j < A[0].size()+B[0].size();j++){
            if(j < A[0].size()){
                res[i][j] = A[i][j];
            }else{
                res[i][j] = B[i][j - A[0].size()];
            }
        }
    }
    return res;
}
Matrix multi_head_attention(const Matrix &X, const Matrix &WQ, const Matrix &WK, const Matrix &WV, vector<Matrix> &Qcut, vector<Matrix> &Kcut, vector<Matrix> &Ktcut, vector<Matrix> &Vcut, vector<Matrix> &scores){
    /*
        Attention = Softmax((Q x Kt)/sqrt(dk)) x V
    */
    int n = X.size();
    int d = X[0].size();
    int dk = WQ[0].size();

    Matrix Q = matmul(X,WQ); // X x WQ
    Matrix K = matmul(X,WK); // X x WK
    Matrix V = matmul(X,WV); // X x WV

    Matrix Kt = transpose(K); // Kt

    Matrix scoresCut;
    Matrix scoresConcat(X.size(), vector<float>(0));
    Matrix head_out;
    Matrix res(X.size(), vector<float>(0));

    for(int m = 0;m < h;m++){
        Matrix Qcut_m(X.size(), vector<float>(dk/h));
        Matrix Kcut_m(X.size(), vector<float>(dk/h));
        Matrix Vcut_m(X.size(), vector<float>(dk/h));
        for(int i = 0;i < Q.size();i++){
            for(int j = 0;j < (Q[0].size())/h;j++){
                Qcut_m[i][j] = Q[i][m*(Q[0].size())/h + j];
                Kcut_m[i][j] = K[i][m*(K[0].size())/h + j];
                Vcut_m[i][j] = V[i][m*(V[0].size())/h + j];
            }
        }
        Matrix Ktcut_m = transpose(Kcut_m);
        scoresCut = matmul(Qcut_m, Ktcut_m);
        //Divided by sqrt(dk) and softmax
        for(int i = 0;i < scoresCut.size();i++){
            for(int j = 0;j < scoresCut[0].size();j++){
                scoresCut[i][j] /= sqrt(dk/h);
            }
        }
        scoresCut = softmax(applyCausalMask(scoresCut));

        scores.push_back(scoresCut);
        Qcut.push_back(Qcut_m);
        Kcut.push_back(Kcut_m);
        Ktcut.push_back(Ktcut_m);
        Vcut.push_back(Vcut_m);

        head_out = matmul(scoresCut, Vcut[m]);

        res = concat(res, head_out);
    }
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
float ReLU(float x){
    return (x >= 0) ? x : 0;
}
Matrix layerNorm(const Matrix &X, const Matrix &gamma, const Matrix &beta, Matrix &Xhat, vector<float> &sigma){
    int n = X.size();
    int d = X[0].size();
    Matrix res(n, vector<float>(d));
    Xhat = Matrix(n, vector<float>(d));
    sigma = vector<float>(n);

    for(int i = 0;i < n;i++){
        float sum = 0.0f;
        for(int j = 0;j < d;j++){
            sum += X[i][j];
        }
        float mean = sum / d;

        sum = 0.0f;
        for(int j = 0;j < d;j++){
            sum += (X[i][j] - mean) * (X[i][j] - mean);
        }
        float var = sum / d;
        
        sigma[i] = sqrt(var + 1e-5);

        for(int j = 0;j < d;j++){
            Xhat[i][j] = (X[i][j] - mean) / sigma[i];
            res[i][j] = gamma[0][j] * Xhat[i][j] + beta[0][j];
        }
    }

    return res;
}
Matrix FNN(const Matrix &X, const Matrix &Wup, const Matrix &Wdown, Matrix &Hact){
    Matrix H = matmul(X, Wup);
    Hact = Matrix(H.size(), vector<float>(H[0].size()));
    for(int i = 0;i < H.size();i++){
        for(int j = 0;j < H[0].size();j++){
            Hact[i][j] = ReLU(H[i][j]);
        }
    }
    Matrix res = matmul(Hact, Wdown);
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
    Vocab_load("train/Vocab.txt", char2id, id2char, vocab_size);
    Matrix tokenEmbedding = Matrix_load("usr/tokenEmbedding.txt");
    Matrix posEmbedding = Matrix_load("usr/posEmbedding.txt");
    Matrix WQ = Matrix_load("train/WQ.txt");
    Matrix WK = Matrix_load("train/WK.txt");
    Matrix WV = Matrix_load("train/WV.txt");
    Matrix Wo = Matrix_load("train/Wo.txt");
    Matrix Wout = Matrix_load("train/Wout.txt");
    Matrix gamma = Matrix_load("train/gamma.txt");
    Matrix beta = Matrix_load("train/beta.txt");
    Matrix Wup = Matrix_load("train/Wup.txt");
    Matrix Wdown = Matrix_load("train/Wdown.txt");
    Matrix gamma2 = Matrix_load("train/gamma2.txt");
    Matrix beta2 = Matrix_load("train/beta2.txt");

    string text = Text_load("train/train.txt");
    vector<int> tokenIds = tokenize(text, char2id);

    vector<int> initialIds = tokenize(input, char2id);
    int remainingSlots = max_seq_len - initialIds.size();

    for(int m = 0;m < remainingSlots;m++){
        /*      Forward     */
        vector<int> window = tokenize(input, char2id);
        Matrix X = matrix_add(embedLookup(window, tokenEmbedding), posEmbedding);

        /*layerNorm 1*/
        Matrix Xhat;
        vector<float> sigma;
        Matrix Xnorm = layerNorm(X, gamma, beta, Xhat, sigma);
        /*Multi-head attention*/
        vector<Matrix> Qcut, Kcut, Ktcut, Vcut, scores;
        Matrix A = multi_head_attention(Xnorm, WQ, WK, WV, Qcut, Kcut, Ktcut, Vcut, scores);
        Matrix AWo = matmul(A, Wo);
        /*Residual Connection 1*/
        Matrix Xrc = matrix_add(X, AWo);
        /*layerNorm 2*/
        Matrix Xhat2;
        vector<float> sigma2;
        Matrix Xnorm2 = layerNorm(Xrc, gamma2, beta2, Xhat2, sigma2);
        /*FNN Connection*/
        Matrix Hact;
        Matrix FNNout = FNN(Xnorm2, Wup, Wdown, Hact);
        /*Residual Connection 2*/
        Matrix Xrc2 = matrix_add(Xrc, FNNout);
        /*Final resulting*/
        Matrix result = matmul(Xrc2, Wout);


        int next = find_next(result);
        cout << id2char[next];
        cout.flush();
        input += id2char[next];
    }
    return 0;
}