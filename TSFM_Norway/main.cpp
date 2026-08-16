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
int dimension = 4 * d;
const int h = 8; //The Multi-head quantity must be checked
const int N = 4;
const int max_seq_len = 120;
const int epoch = 10000;
const int record = 500;

struct BlockParams{
    Matrix WQ, WK, WV, Wo;
    Matrix gamma, beta, gamma2, beta2;
    Matrix Wup, Wdown;
};

struct BlockCache{
    Matrix Xhat;
    vector<float> sigma;
    Matrix Xnorm;
    vector<float> Qcut, Kcut, Ktcut, Vcut, scores;
    Matrix A, Xrc;
    Matrix Xhat2;
    vector<float> sigma2;
    Matrix Xnorm2, Hact, Xrc2;
};

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
/*      Backward Function       */
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
            res[i][j] = dCM[i][j] / sqrt(dk/h);
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
void save(const Matrix &WQ, const Matrix &WK, const Matrix &WV, const Matrix &Wo, const Matrix &Wout, const Matrix &tokenEmbedding, const Matrix &posEmbedding, float loss, const Matrix &gamma, const Matrix &beta, const Matrix &gamma2, const Matrix &beta2, const Matrix &Wup, const Matrix &Wdown){
    ofstream tokenEmbedder("train/tokenEmbedding.txt");
    ofstream posEmbedder("train/posEmbedding.txt");
    ofstream WQr("train/WQ.txt");
    ofstream WKr("train/WK.txt");
    ofstream WVr("train/WV.txt");
    ofstream Wor("train/Wo.txt");
    ofstream Wouter("train/Wout.txt");
    ofstream Losser("train/Loss.txt",ios::app);
    ofstream gammar("train/gamma.txt");
    ofstream betar("train/beta.txt");
    ofstream gammar2("train/gamma2.txt");
    ofstream betar2("train/beta2.txt");
    ofstream Wupr("train/Wup.txt");
    ofstream Wdownr("train/Wdown.txt");

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
    Wor << d <<' ' << dk <<endl;
    for(int i = 0;i < d;i++){
        for(int j = 0;j < dk;j++){
            WQr << WQ[i][j] << ' ';
            WKr << WK[i][j] << ' ';
            WVr << WV[i][j] << ' ';
            Wor << Wo[i][j] << ' ';
        }
        WQr <<endl;
        WKr <<endl;
        WVr <<endl;
        Wor <<endl;
    }
    WQr.close();
    WKr.close();
    WVr.close();
    Wor.close();

    Wouter << d <<' ' << vocab_size <<endl;
    for(int i = 0;i < d;i++){
        for(int j = 0;j < vocab_size;j++){
            Wouter << Wout[i][j] <<' ';
        }
        Wouter <<endl;
    }
    Wouter.close();

    Losser << loss <<endl;
    Losser.close();

    gammar << 1 << ' ' << d <<endl;
    betar << 1 << ' ' << d <<endl;
    for(int i = 0;i < d;i++){
        gammar << gamma[0][i] << ' ';
        betar << beta[0][i] << ' ';
    }
    gammar.close();
    betar.close();

    gammar2 << 1 << ' ' << d <<endl;
    betar2 << 1 << ' ' << d <<endl;
    for(int i = 0;i < d;i++){
        gammar2 << gamma2[0][i] << ' ';
        betar2 << beta2[0][i] << ' ';
    }
    gammar2.close();
    betar2.close();

    Wupr << d << ' ' << dimension <<endl;
    Wdownr << dimension << ' ' << d <<endl;
    for(int i = 0;i < d;i++){
        for(int j = 0;j < dimension;j++){
            Wupr << Wup[i][j] << ' ';
        }
        Wupr << endl;
    }
    for(int i = 0;i < dimension;i++){
        for(int j = 0;j < d;j++){
            Wdownr << Wdown[i][j] << ' ';
        }
        Wdownr <<endl;
    }
    Wupr.close();
    Wdownr.close();
}
void layerNorm_backward(const Matrix &dXnorm, const Matrix &Xhat, const vector<float> &sigma, const Matrix &gamma, Matrix &dXnoXrc, Matrix &dgamma, Matrix &dbeta){
    int n = dXnorm.size();
    int d = dXnorm[0].size();

    Matrix dXhat(n, vector<float>(d));
    dXnoXrc = Matrix(n, vector<float>(d));
    dgamma = Matrix(1, vector<float>(d, 0.0f));
    dbeta = Matrix(1, vector<float>(d, 0.0f));
    for(int i = 0;i < n;i++){
        for(int j = 0;j < d;j++){
            dXhat[i][j] = dXnorm[i][j] * gamma[0][j];
        }

        float sum = 0.0f;
        for(int j = 0;j < d;j++){
            sum += dXhat[i][j];
        }
        float mean_dXhat = sum / d;

        sum = 0.0f;
        for(int j = 0;j < d;j++){
            sum += dXhat[i][j] * Xhat[i][j];
        }
        float mean_dXhat_Xhat = sum / d;

        sum = 0.0f;
        for(int j = 0;j < d;j++){
            dXnoXrc[i][j] = (dXhat[i][j] - mean_dXhat - Xhat[i][j] * mean_dXhat_Xhat) / sigma[i];
            dgamma[0][j] += dXnorm[i][j] * Xhat[i][j];
            dbeta[0][j] += dXnorm[i][j];
        }
    }
}
void FNN_backward(const Matrix &dFNNout, const Matrix &Xnorm2, const Matrix &Hact, const Matrix &Wup, const Matrix &Wdown, Matrix &dXnorm2, Matrix &dWup, Matrix &dWdown){
    Matrix dHact;
    matmul_backward(Hact, Wdown, dFNNout, dHact, dWdown);

    Matrix dH(dHact.size(), vector<float>(dHact[0].size()));
    for(int i = 0;i < dH.size();i++){
        for(int j = 0;j < dH[0].size();j++){
            dH[i][j] = (Hact[i][j] > 0) ? dHact[i][j] : 0;
        }
    }

    matmul_backward(Xnorm2, Wup, dH, dXnorm2, dWup);
}
int main()
{
    random_device rd;
    mt19937 gen(rd());

    vector<BlockParams> layers(N);
    vector<BlockCache> caches(N);
    /*      Loading data     */
    map<string, int> char2id;
    vector<string> id2char;
    Vocab_load("train/Vocab.txt", char2id, id2char, vocab_size);
    Matrix tokenEmbedding = Matrix_load("train/tokenEmbedding.txt");
    Matrix posEmbedding = Matrix_load("train/posEmbedding.txt");
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

    int r = 1;
    for(int e = 0;e < epoch;e++){
        /*      Forward     */
        vector<int> window = window_select(tokenIds, gen);
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


        float loss = losslize(result, window);
        cout << "LOSS:" <<loss/(max_seq_len-1) << "][Epoch:" << e << "/"<< epoch<<endl;

        /*      Backward        */

        /*Softmax backward*/
        Matrix grad_dC;
        softmax_crossentropy_backward(result, window, grad_dC);

        /*result / Residual connection/AWo backward*/
        Matrix grad_dXrc2, grad_dWout;
        matmul_backward(Xrc2, Wout, grad_dC, grad_dXrc2, grad_dWout);
        
        /*FNN backward*/
        Matrix grad_dFNNout = grad_dXrc2;
        Matrix grad_dXnorm2, grad_dWup, grad_dWdown;
        FNN_backward(grad_dFNNout, Xnorm2, Hact, Wup, Wdown, grad_dXnorm2, grad_dWup, grad_dWdown);

        /*layerNorm 2 backward*/
        Matrix grad_dXnoXrc2, grad_dgamma2, grad_dbeta2;
        layerNorm_backward(grad_dXnorm2, Xhat2, sigma2, gamma2, grad_dXnoXrc2, grad_dgamma2, grad_dbeta2);
        Matrix grad_dXrc = matrix_add(grad_dXnoXrc2, grad_dXrc2);

        /*Multi-head attention backward*/
        Matrix grad_dAWo = grad_dXrc;
        Matrix grad_dA, grad_dWo;
        matmul_backward(A, Wo, grad_dAWo, grad_dA, grad_dWo);

        vector<Matrix> grad_dAcut;
        for(int m = 0;m < h;m++){
            Matrix grad_dAcut_m(grad_dA.size(), vector<float>((grad_dA[0].size()/h)));
            for(int i = 0;i < grad_dA.size();i++){
                for(int j = 0;j < (grad_dA[0].size()/h);j++){
                    grad_dAcut_m[i][j] = grad_dA[i][m*(grad_dA[0].size())/h+j];
                }
            }
            grad_dAcut.push_back(grad_dAcut_m);
        }
        Matrix grad_scores, grad_dV;
        Matrix grad_sftmx_scores;
        Matrix grad_dCM;
        Matrix grad_dS;
        Matrix grad_dQ, grad_dKt;
        matmul_backward(scores[0], Vcut[0], grad_dAcut[0], grad_scores, grad_dV);
        softmax_attention_backward(scores[0], grad_scores, grad_sftmx_scores);
        applyCausalMask_backward(grad_sftmx_scores, grad_dCM);
        scale_backward(grad_dCM, grad_dS);
        matmul_backward(Qcut[0], Ktcut[0], grad_dS, grad_dQ, grad_dKt);
        Matrix grad_dK = transpose(grad_dKt);
        for(int m = 1;m < h;m++){
            /*Attention Scores/V backward*/
            Matrix grad_scores_m, grad_dV_m;
            matmul_backward(scores[m], Vcut[m], grad_dAcut[m], grad_scores_m, grad_dV_m);
            /*Attention Softmax backward*/
            Matrix grad_sftmx_scores_m;
            softmax_attention_backward(scores[m], grad_scores_m, grad_sftmx_scores_m);
            /*Attention CM backward*/
            Matrix grad_dCM_m;
            applyCausalMask_backward(grad_sftmx_scores_m, grad_dCM_m);
            /*Attention Scale backward*/
            Matrix grad_dS_m;
            scale_backward(grad_dCM_m, grad_dS_m);
            /*Attention Q/K backward*/
            Matrix grad_dQ_m, grad_dKt_m;
            matmul_backward(Qcut[m], Ktcut[m], grad_dS_m, grad_dQ_m, grad_dKt_m);
            Matrix grad_dK_m = transpose(grad_dKt_m);

            grad_scores = concat(grad_scores, grad_scores_m);
            grad_dV = concat(grad_dV, grad_dV_m);
            grad_sftmx_scores = concat(grad_sftmx_scores, grad_sftmx_scores_m);
            grad_dCM = concat(grad_dCM, grad_dCM_m);
            grad_dS = concat(grad_dS, grad_dS_m);
            grad_dQ = concat(grad_dQ, grad_dQ_m);
            grad_dKt = concat(grad_dKt, grad_dKt_m);
            grad_dK = concat(grad_dK, grad_dK_m);
        }
        /*Attention WQKV backward*/
        Matrix grad_dWQ, grad_dWK, grad_dWV, grad_dXQ, grad_dXK, grad_dXV;
        matmul_backward(Xnorm, WQ, grad_dQ, grad_dXQ, grad_dWQ);
        matmul_backward(Xnorm, WK, grad_dK, grad_dXK, grad_dWK);
        matmul_backward(Xnorm, WV, grad_dV, grad_dXV, grad_dWV);
        Matrix grad_dXnorm = matrix_add(matrix_add(grad_dXQ, grad_dXK), grad_dXV);

        /*layerNorm 1 backward*/
        Matrix grad_dXnoXrc, grad_dgamma, grad_dbeta;
        layerNorm_backward(grad_dXnorm, Xhat, sigma, gamma, grad_dXnoXrc, grad_dgamma, grad_dbeta);

        Matrix grad_dX = matrix_add(grad_dXnoXrc, grad_dXrc);
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
        clip(grad_dWo, clipLimit);
        clip(grad_dWout, clipLimit);
        clip(grad_dgamma, clipLimit);
        clip(grad_dbeta, clipLimit);
        clip(grad_dgamma2, clipLimit);
        clip(grad_dbeta2, clipLimit);
        clip(grad_dWup, clipLimit);
        clip(grad_dWdown, clipLimit);
        clip(grad_dwindow, clipLimit);
        clip(grad_posTable_partial, clipLimit);

        update(WQ, grad_dWQ, lr);
        update(WK, grad_dWK, lr);
        update(WV, grad_dWV, lr);
        update(Wo, grad_dWo, lr);
        update(Wout, grad_dWout, lr);
        update(gamma, grad_dgamma, lr);
        update(beta, grad_dbeta, lr);
        update(gamma2, grad_dgamma2, lr);
        update(beta2, grad_dbeta2, lr);
        update(Wup, grad_dWup, lr);
        update(Wdown, grad_dWdown, lr);
        update(tokenEmbedding, grad_dwindow, lr);
        update_partial(posEmbedding, grad_posTable_partial, lr);

        if(r == record){
            save(WQ, WK, WV, Wo, Wout, tokenEmbedding, posEmbedding, loss/(max_seq_len-1), gamma, beta, gamma2, beta2, Wup, Wdown);
            r = 0;
        }
        r++;
    }
    return 0;
}