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
Matrix matrix_add(const Matrix &A, const Matrix &B){
    Matrix res(A.size(), vector<float>(A[0].size()));
    for(int i = 0;i < A.size();i++){
        for(int j = 0;j < A[0].size();j++){
            res[i][j] = A[i][j] + B[i][j];
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

float losslize(const Matrix &X, const vector<int> &window){
    Matrix sftmx = softmax(X);
    float loss = 0.0f;
    int n = sftmx.size();
    for(int i = 0;i < n - 1;i++){
        int ans = window[i+1];
        loss += -log(sftmx[i][ans]);
    }
    return loss;
}
void softmax_crossentropy_backward(const Matrix &X, const vector<int> &window, Matrix &dLogits){
    Matrix res(X.size(), vector<float>(X[0].size()));
    Matrix sftmx = softmax(X);
    int n = sftmx.size();
    for(int i = 0;i < n - 1;i++){
        int ans = window[i+1];
        res[i] = sftmx[i];
        res[i][ans] -= 1;
    }
    dLogits = res;
}
void matmul_backward(const Matrix &A, const Matrix &B, const Matrix &dC, Matrix &dA, Matrix &dB){
    dA = matmul(dC, transpose(B));
    dB = matmul(transpose(A), dC);
}
void softmax_attention_backward(const Matrix &scores, const Matrix &grad_scores, Matrix &sftmx){
    Matrix res(scores.size(), vector<float>(scores[0].size()));
    for(int i = 0;i < scores.size();i++){
        float sum = 0.0f;
        for(int j = 0;j < scores[0].size();j++){
            sum += grad_scores[i][j] * scores[i][j];
        }
        for(int j = 0;j < scores[0].size();j++){
            res[i][j] = scores[i][j] * (grad_scores[i][j] - sum);
        }
    }
    sftmx = res;
}
void applyCausalMask_backward(const Matrix &grad_scores, Matrix &dCM){
    int n = grad_scores.size();
    Matrix res(n, vector<float>(n, 0.0f));

    for(int i = 0;i < n;i++){
        for(int j = 0;j <= i;j++){
            res[i][j] = grad_scores[i][j];
        }
    }
    dCM = res;
}
void scale_backward(const Matrix &dCM, Matrix &dS){
    Matrix res(dCM.size(), vector<float>(dCM[0].size()));
    for(int i = 0;i < dCM.size();i++){
        for(int j = 0;j < dCM[0].size();j++){
            res[i][j] = dCM[i][j] / sqrt(dk);
        }
    }
    dS = res;
}

void embedLookup_backward(const vector<int> &window, const Matrix &tokenVecs, Matrix &dwindow){
    int d = tokenVecs[0].size();
    Matrix res(vocab_size, vector<float>(d, 0.0f));

    for(int i = 0;i < window.size();i++){
        int id = window[i];
        for(int j = 0;j < d;j++){
            res[id][j] += tokenVecs[i][j];
        }
    }
    dwindow = res;
}
void update(Matrix &W, const Matrix &grad, float lr){
    for(int i = 0;i < W.size();i++){
        for(int j = 0;j < W[0].size();j++){
            W[i][j]  = W[i][j] - lr * grad[i][j];
        }
    }
}
void update_partial(Matrix &W, const Matrix &grad, float lr){
    for(int i = 0;i < grad.size();i++){
        for(int j = 0;j < grad[0].size();j++){
            W[i][j] = W[i][j] - lr * grad[i][j];
        }
    }
}
vector<int> window_select(const vector<int> &tokenIds, mt19937 &gen){
    int total = tokenIds.size();
    int winLen = min(max_seq_len, total);
    int maxStart = total - winLen;
    uniform_int_distribution<int> dist(0, maxStart);
    int start = dist(gen);
    return vector<int>(tokenIds.begin() + start, tokenIds.begin() + start + winLen);
}
void clip(Matrix &grad, float limit){
    for(int i = 0;i < (int)grad.size();i++){
        for(int j = 0;j < (int)grad[0].size();j++){
            if(grad[i][j] > limit) grad[i][j] = limit;
            if(grad[i][j] < -limit) grad[i][j] = -limit;
        }
    }
}
void save(const Matrix &WQ, const Matrix &WK, const Matrix &WV, const Matrix &Wout, const Matrix &tokenEmbedding, const Matrix &posEmbedding){
    ofstream tokenEmbedder("train/tokenEmbedding.txt");
    ofstream posEmbedder("train/posEmbedding.txt");
    ofstream WQr("train/WQ.txt");
    ofstream WKr("train/WK.txt");
    ofstream WVr("train/WV.txt");
    ofstream Wouter("train/Wout.txt");

    tokenEmbedder << vocab_size << ' '<< d <<endl;
    for(int i = 0;i < vocab_size;i++){
        for(int j = 0;j < d;j++){
            tokenEmbedder << tokenEmbedding[i][j] <<' ';
        }
        tokenEmbedder <<endl;
    }
    tokenEmbedder.close();

    posEmbedder << max_seq_len << ' '<< d <<endl;
    for(int i = 0;i < max_seq_len;i++){
        for(int j = 0;j < d;j++){
            posEmbedder << posEmbedding[i][j] <<' ';
        }
        posEmbedder <<endl;
    }
    posEmbedder.close();

    WQr << d <<' ' << dk <<endl;
    WKr << d <<' ' << dk <<endl;
    WVr << d <<' ' << dk <<endl;
    for(int i = 0;i < d;i++){
        for(int j = 0;j < dk;j++){
            WQr << WQ[i][j] << ' ';
            WKr << WK[i][j] << ' ';
            WVr << WV[i][j] << ' ';
        }
        WQr <<endl;
        WKr <<endl;
        WVr <<endl;
    }
    WQr.close();
    WKr.close();
    WVr.close();

    Wouter << d <<' ' << vocab_size <<endl;
    for(int i = 0;i < d;i++){
        for(int j = 0;j < vocab_size;j++){
            Wouter << Wout[i][j] <<' ';
        }
        Wouter <<endl;
    }
    Wouter.close();
}
int main()
{
    random_device rd;
    mt19937 gen(rd());

    /*      Loading data     */
    map<string, int> char2id;
    vector<string> id2char;
    Vocab_load("train/Vocab.txt", char2id, id2char, vocab_size);
    Matrix tokenEmbedding = Matrix_load("train/tokenEmbedding.txt");
    Matrix posEmbedding = Matrix_load("train/posEmbedding.txt");
    Matrix WQ = Matrix_load("train/WQ.txt");
    Matrix WK = Matrix_load("train/WK.txt");
    Matrix WV = Matrix_load("train/WV.txt");
    Matrix Wout = Matrix_load("train/Wout.txt");

    string text = Text_load("train/train.txt");
    vector<int> tokenIds = tokenize(text, char2id);

    int r = 1;
    for(int e = 0;e < epoch;e++){
        /*      Forward     */
        vector<int> window = window_select(tokenIds, gen);
        
        Matrix X = matrix_add(embedLookup(window, tokenEmbedding), posEmbedding);
        Matrix Q, K, Kt, V, scores;
        Matrix A = attention(X, WQ, WK, WV, Q, K, Kt, V, scores);
        Matrix result = matmul(A, Wout);

        float loss = losslize(result, window);
        cout << "LOSS:" <<loss/(max_seq_len-1) << "][Epoch:" << e << "/"<< epoch<<endl;

        /*      Backward        */

        /*Softmax backward*/
        Matrix grad_dC;
        softmax_crossentropy_backward(result, window, grad_dC);
        /*result/Residual connection/AWo backward*/
        Matrix grad_dA, grad_dWout;
        matmul_backward(A, Wout, grad_dC, grad_dA, grad_dWout);
        /*Attention Scores/V backward*/
        Matrix grad_scores, grad_dV;
        matmul_backward(scores, V, grad_dA, grad_scores, grad_dV);
        /*Attention Softmax backward*/
        Matrix grad_sftmx_scores;
        softmax_attention_backward(scores, grad_scores, grad_sftmx_scores);
        /*Attention CM backward*/
        Matrix grad_dCM;
        applyCausalMask_backward(grad_sftmx_scores, grad_dCM);
        /*Attention Scale backward*/
        Matrix grad_dS;
        scale_backward(grad_dCM, grad_dS);
        /*Attention Q/K backward*/
        Matrix grad_dQ, grad_dKt;
        matmul_backward(Q, Kt, grad_dS, grad_dQ, grad_dKt);
        Matrix grad_dK = transpose(grad_dKt);
        /*WQKV backward*/
        Matrix grad_dWQ, grad_dWK, grad_dWV, grad_dXQ, grad_dXK, grad_dXV;
        matmul_backward(X, WQ, grad_dQ, grad_dXQ, grad_dWQ);
        matmul_backward(X, WK, grad_dK, grad_dXK, grad_dWK);
        matmul_backward(X, WV, grad_dV, grad_dXV, grad_dWV);
        Matrix grad_dX = matrix_add(matrix_add(grad_dXQ, grad_dXK), grad_dXV);
        /*Encoding backward*/
        Matrix grad_tokenVecs = grad_dX;
        Matrix grad_posTable_partial = grad_dX;
        /*Embedding backward*/
        Matrix grad_dwindow;
        embedLookup_backward(window, grad_tokenVecs, grad_dwindow);

        /*      Machine Learning        */

        float lr = 0.001f;
        float clipLimit = 1.0f;

        clip(grad_dWQ, clipLimit);
        clip(grad_dWK, clipLimit);
        clip(grad_dWV, clipLimit);
        clip(grad_dWout, clipLimit);
        clip(grad_dwindow, clipLimit);
        clip(grad_posTable_partial, clipLimit);

        update(WQ, grad_dWQ, lr);
        update(WK, grad_dWK, lr);
        update(WV, grad_dWV, lr);
        update(Wout, grad_dWout, lr);
        update(tokenEmbedding, grad_dwindow, lr);
        update_partial(posEmbedding, grad_posTable_partial, lr);

        if(r == record){
            save(WQ, WK, WV, Wout, tokenEmbedding, posEmbedding);
            r = 0;
        }
        r++;
    }
    return 0;
}