#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <set>
#include <direct.h>
#include <windows.h>
#include <cuda_runtime.h>
#define M_PI 3.1415926
using namespace std;

typedef vector<vector<float>> Matrix;
int vocab_size = 0, d = 384, dk = 384;
int dimension = 4 * d;
int h = 6; //The Multi-head quantity must be checked
int N = 8;
int max_seq_len = 256;
int epoch = 10000;
const int record = 1000;
/*  Real Tokens   */
vector<string> id2token;
unordered_map<string, int> token2id;
map<pair<int, int>, int> pairRank;
vector<pair<int, int>> rankPair;
vector<int> rankNew;
int UNK = 0;
int lossFrom = 1;
vector<string> specialTokens;



struct BlockParams {
    Matrix WQ, WK, WV, Wo;
    Matrix gamma, beta, gamma2, beta2;
    Matrix Wup, Wdown;
};

struct BlockCache {
    Matrix Xhat;
    vector<float> sigma;
    Matrix Xnorm;
    vector<Matrix> Qcut, Kcut, Ktcut, Vcut, scores;
    Matrix A, Xrc;
    Matrix Xhat2;
    vector<float> sigma2;
    Matrix Xnorm2, Hact, Xrc2;
};

struct BlockParamGrads {
    Matrix grad_dWQ, grad_dWK, grad_dWV, grad_dWo;
    Matrix grad_dgamma, grad_dbeta, grad_dgamma2, grad_dbeta2;
    Matrix grad_dWup, grad_dWdown;
};
struct AdamState {
    Matrix m, v;
};
struct BlockAdamState {
    AdamState WQ, WK, WV, Wo;
    AdamState gamma, beta, gamma2, beta2;
    AdamState Wup, Wdown;
};


/* ==================== CUDA ==================== */

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            abort();                                                            \
        }                                                                       \
    } while (0)

#define TILE 16

// 逻辑上算 C[M][N] = op(A)[M][K] * op(B)[K][N]，全部 row-major。
// tA=1 表示 A 物理是 K×M（需要转置读），tB=1 表示 B 物理是 N×K。
__global__ void matmul_kernel(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N, int tA, int tB)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;   // 对应原来的 gid.y
    int col = blockIdx.x * TILE + threadIdx.x;   // 对应原来的 gid.x

    float sum = 0.0f;
    int nTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < nTiles; ++t) {
        int aK = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (row < M && aK < K)
            ? (tA ? A[(size_t)aK * M + row] : A[(size_t)row * K + aK])
            : 0.0f;

        int bK = t * TILE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] =
            (col < N && bK < K)
            ? (tB ? B[(size_t)col * K + bK] : B[(size_t)bK * N + col])
            : 0.0f;

        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE; ++i)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[(size_t)row * N + col] = sum;
}

void initCUDA() {
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        fprintf(stderr, "没有找到 CUDA 设备\n");
        abort();
    }
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("CUDA device: %s (SM %d.%d, %d SMs)\n",
        prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    CUDA_CHECK(cudaFree(0));   // 提前建立 context，避免第一次 matmul 被初始化开销污染
}

// 按需扩容的一段 buffer：设备端 + 主机端 pinned 暂存
struct GpuBuf {
    float* dev = nullptr;
    float* host = nullptr;   // pinned，用于 H2D / D2H 暂存
    size_t cap = 0;         // 字节
};

static void ensureCap(GpuBuf& b, size_t bytes) {
    if (b.cap >= bytes) return;
    if (b.dev)  CUDA_CHECK(cudaFree(b.dev));
    if (b.host) CUDA_CHECK(cudaFreeHost(b.host));
    CUDA_CHECK(cudaMalloc(&b.dev, bytes));
    CUDA_CHECK(cudaHostAlloc(&b.host, bytes, cudaHostAllocDefault));
    b.cap = bytes;
}

Matrix matmul(const Matrix& A, const Matrix& B, bool tA = false, bool tB = false) {
    static GpuBuf bufA, bufB, bufC;

    // A 物理尺寸 aR×aC，逻辑尺寸 M×K
    int aR = A.size(), aC = A[0].size();
    int bR = B.size(), bC = B[0].size();
    int M = tA ? aC : aR;
    int K = tA ? aR : aC;
    int N = tB ? bR : bC;
    int Kb = tB ? bC : bR;

    if (K != Kb) {
        fprintf(stderr, "matmul 维度不匹配: A(%d,%d,tA=%d) B(%d,%d,tB=%d)\n",
            aR, aC, (int)tA, bR, bC, (int)tB);
        abort();
    }

    size_t szA = (size_t)aR * aC * sizeof(float);
    size_t szB = (size_t)bR * bC * sizeof(float);
    size_t szC = (size_t)M * N * sizeof(float);

    ensureCap(bufA, szA);
    ensureCap(bufB, szB);
    ensureCap(bufC, szC);

    // 按物理形状原样打平，转置交给 kernel
    for (int i = 0; i < aR; i++)
        memcpy(bufA.host + (size_t)i * aC, A[i].data(), aC * sizeof(float));
    for (int i = 0; i < bR; i++)
        memcpy(bufB.host + (size_t)i * bC, B[i].data(), bC * sizeof(float));

    CUDA_CHECK(cudaMemcpy(bufA.dev, bufA.host, szA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bufB.dev, bufB.host, szB, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_kernel << <grid, block >> > (bufA.dev, bufB.dev, bufC.dev,
        M, K, N, tA ? 1 : 0, tB ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(bufC.host, bufC.dev, szC, cudaMemcpyDeviceToHost));

    Matrix res(M, vector<float>(N));
    for (int i = 0; i < M; i++)
        memcpy(res[i].data(), bufC.host + (size_t)i * N, N * sizeof(float));
    return res;
}

/* ================== CUDA 结束 ================== */

Matrix transpose(const Matrix& A) {
    int Arow = A.size();
    int Acolumn = A[0].size();
    Matrix res(Acolumn, vector<float>(Arow, 0.0f));

    for (int i = 0;i < Arow; i++) {
        for (int j = 0;j < Acolumn;j++) {
            res[j][i] = A[i][j];
        }
    }

    return res;
}
Matrix softmax(const Matrix& A) {
    int Arow = A.size();
    int Acolumn = A[0].size();
    Matrix res(Arow, vector<float>(Acolumn, 0.0f));

    for (int i = 0;i < Arow;i++) {
        float e_sum = 0.0f;
        float maxv = A[i][0];
        for (int j = 0;j < Acolumn;j++) {
            maxv = max(A[i][j], maxv);
        }
        for (int j = 0;j < Acolumn;j++) {
            e_sum += exp(A[i][j] - maxv);
        }
        for (int j = 0;j < Acolumn;j++) {
            res[i][j] = exp(A[i][j] - maxv) / e_sum;
        }
    }

    return res;
}
Matrix applyCausalMask(const Matrix& CM) {
    int n = CM.size(); //Width and Height are the same size
    Matrix res(n, vector<float>(n, -1e9f));

    for (int i = 0;i < n;i++) {
        for (int j = 0;j <= i;j++) {
            res[i][j] = CM[i][j];
        }
    }
    return res;
}
Matrix matrix_add(const Matrix& A, const Matrix& B) {
    Matrix res(A.size(), vector<float>(A[0].size()));
    for (int i = 0;i < A.size();i++) {
        for (int j = 0;j < A[0].size();j++) {
            res[i][j] = A[i][j] + B[i][j];
        }
    }
    return res;
}
Matrix concat(const Matrix& A, const Matrix& B) {
    Matrix res(A.size(), vector<float>(A[0].size() + B[0].size()));
    for (int i = 0;i < A.size();i++) {
        for (int j = 0;j < A[0].size() + B[0].size();j++) {
            if (j < A[0].size()) {
                res[i][j] = A[i][j];
            }
            else {
                res[i][j] = B[i][j - A[0].size()];
            }
        }
    }
    return res;
}
Matrix multi_head_attention(const Matrix& X, const Matrix& WQ, const Matrix& WK, const Matrix& WV, vector<Matrix>& Qcut, vector<Matrix>& Kcut, vector<Matrix>& Ktcut, vector<Matrix>& Vcut, vector<Matrix>& scores) {
    /*
        Attention = Softmax((Q x Kt)/sqrt(dk)) x V
    */
    int n = X.size();
    int d = X[0].size();
    int dk = WQ[0].size();

    Matrix Q = matmul(X, WQ); // X x WQ
    Matrix K = matmul(X, WK); // X x WK
    Matrix V = matmul(X, WV); // X x WV

    Matrix Kt = transpose(K); // Kt

    Matrix scoresCut;
    Matrix scoresConcat(X.size(), vector<float>(0));
    Matrix head_out;
    Matrix res(X.size(), vector<float>(0));

    for (int m = 0;m < h;m++) {
        Matrix Qcut_m(X.size(), vector<float>(dk / h));
        Matrix Kcut_m(X.size(), vector<float>(dk / h));
        Matrix Vcut_m(X.size(), vector<float>(dk / h));
        for (int i = 0;i < Q.size();i++) {
            for (int j = 0;j < (Q[0].size()) / h;j++) {
                Qcut_m[i][j] = Q[i][m * (Q[0].size()) / h + j];
                Kcut_m[i][j] = K[i][m * (K[0].size()) / h + j];
                Vcut_m[i][j] = V[i][m * (V[0].size()) / h + j];
            }
        }
        Matrix Ktcut_m = transpose(Kcut_m);
        scoresCut = matmul(Qcut_m, Ktcut_m);
        //Divided by sqrt(dk) and softmax
        for (int i = 0;i < scoresCut.size();i++) {
            for (int j = 0;j < scoresCut[0].size();j++) {
                scoresCut[i][j] /= sqrt(dk / h);
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
Matrix Matrix_load(string filename) {
    ifstream file(filename);
    int a, b;
    file >> a >> b;

    Matrix res(a, vector<float>(b));
    for (int i = 0;i < a;i++) {
        for (int j = 0;j < b;j++) {
            file >> res[i][j];
        }
    }
    return res;
}
string Text_load(string filename) {
    ifstream file(filename);
    stringstream buffer;
    buffer << file.rdbuf();
    string text = buffer.str();
    return text;
}
bool isBoundary(const std::string& x) {
    static const std::set<std::string> b = {
        // 中文标点（全角）
        "\u3002",  // 。 句号
        "\uff0c",  // ， 逗号
        "\u3001",  // 、 顿号
        "\uff1b",  // ； 分号
        "\uff1a",  // ： 冒号
        "\uff1f",  // ？ 问号
        "\uff01",  // ！ 叹号
        "\u201c",  // “ 左双引号
        "\u201d",  // ” 右双引号
        "\u2018",  // ‘ 左单引号
        "\u2019",  // ’ 右单引号
        "\uff08",  // （ 左圆括号
        "\uff09",  // ） 右圆括号
        "\u3010",  // 【 左方括号
        "\u3011",  // 】 右方括号
        "\u300a",  // 《 左书名号
        "\u300b",  // 》 右书名号
        "\u2014",  // — 破折号
        "\u2026",  // … 省略号
        "\u3000",  // 　 全角空格

        // 空白符
        "\n", "\r", " ",

        // 英文标点（保留原有逻辑）
        ".", ",", ";", ":", "?", "!"
    };
    return b.count(x) > 0;
}

void split(const string& x, vector<vector<string>>& corpus) {
    int len = 0;
    vector<string> cur;
    for (size_t i = 0;i < x.size();i += len) {
        unsigned char t = x[i];
        if (t < 128) len = 1;
        else if (t < 224) len = 2;
        else if (t < 240) len = 3;
        else len = 4;

        string byte = x.substr(i, len);

        if (!isBoundary(byte)) {
            cur.push_back(byte);
        }
        else {
            if (!cur.empty()) {
                corpus.push_back(cur);
                cur.clear();
            }
            if (byte != " " && byte != "\n" && byte != "\r" && byte != "銆€") {
                corpus.push_back({ byte });
            }
        }
    }
    if (!cur.empty()) {
        corpus.push_back(cur);
    }
}
void BPE_load(const string& vocabFile, const string& mergeFile) {
    ifstream f(vocabFile);
    int size;
    f >> size;
    f.ignore();
    for (int i = 0;i < size;i++) {
        string line;
        getline(f, line);
        size_t sp = line.find(' ');
        id2token.push_back(line.substr(sp + 1));
        token2id[line.substr(sp + 1)] = i;
    }
    f.close();

    ifstream g(mergeFile);
    int m;
    g >> m;
    for (int i = 0;i < m;i++) {
        int a, b, nid;
        g >> a >> b >> nid;
        pairRank[{a, b}] = i;
        rankPair.push_back({ a,b });
        rankNew.push_back(nid);
    }
    g.close();

    vocab_size = id2token.size();
    UNK = token2id["<|unk|>"];
    for (int i = 0;i < (int)id2token.size();i++) {
        const string& s = id2token[i];
        if (s.size() >= 4 && s.substr(0, 2) == "<|" && s.substr(s.size() - 2) == "|>") {
            specialTokens.push_back(s);
        }
    }
}
vector<int> encodeSegment(vector<int> seg) {
    while (true) {
        int bestRank = -1;
        for (int i = 0;i + 1 < (int)seg.size();i++) {
            map<pair<int, int>, int>::iterator it = pairRank.find({ seg[i], seg[i + 1] });
            if (it == pairRank.end()) continue;
            if (bestRank < 0 || it->second < bestRank) bestRank = it->second;
        }
        if (bestRank < 0)break;

        int a = rankPair[bestRank].first, b = rankPair[bestRank].second;
        int newId = rankNew[bestRank];
        vector<int> out;
        for (int i = 0;i < (int)seg.size();) {
            if (i + 1 < (int)seg.size() && seg[i] == a && seg[i + 1] == b) {
                out.push_back(newId);
                i += 2;
            }
            else {
                out.push_back(seg[i]);
                i += 1;
            }
        }
        seg = out;
    }
    return seg;
}
vector<int> tokenizePlain(const string& text) {
    vector<vector<string>> segsStr;
    split(text, segsStr);
    vector<int> res;
    for (int i = 0;i < (int)segsStr.size();i++) {
        vector<int> seg;
        for (int j = 0;j < (int)segsStr[i].size();j++) {
            unordered_map<string, int>::iterator it = token2id.find(segsStr[i][j]);
            seg.push_back(it != token2id.end() ? it->second : UNK);
        }
        seg = encodeSegment(seg);
        for (int j = 0;j < (int)seg.size();j++) {
            res.push_back(seg[j]);
        }
    }
    return res;
}
vector<int> tokenize(const string& text) {
    vector<int> res;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t best = string::npos, bestLen = 0;
        int bestId = -1;
        for (int k = 0;k < (int)specialTokens.size();k++) {
            size_t p = text.find(specialTokens[k], pos);
            if (p == string::npos)continue;
            if (p < best || (p == best && specialTokens[k].size() > bestLen)) {
                best = p;
                bestLen = specialTokens[k].size();
                bestId = token2id[specialTokens[k]];
            }
        }
        size_t end = (best == string::npos) ? text.size() : best;
        if (end > pos) {
            vector<int> part = tokenizePlain(text.substr(pos, end - pos));
            res.insert(res.end(), part.begin(), part.end());
        }
        if (best == string::npos)break;
        res.push_back(bestId);
        pos = best + bestLen;
    }
    return res;
}
Matrix embedLookup(const vector<int>& window, const Matrix& tokenTable) {
    Matrix res(window.size(), vector<float>(tokenTable[0].size()));

    for (int i = 0;i < window.size();i++) {
        res[i] = tokenTable[window[i]];
    }
    return res;
}

float losslize(const Matrix& X, const vector<int>& window) {
    Matrix sftmx = softmax(X);
    float loss = 0.0f;
    int n = sftmx.size(), cnt = 0;
    for (int i = 0;i < n - 1;i++) {
        if (i + 1 < lossFrom) continue;
        int ans = window[i + 1];
        loss += -log(sftmx[i][ans]);
        cnt++;
    }
    return cnt ? loss / (float)cnt : 0.0f;
}
float ReLU(float x) {
    return (x >= 0) ? x : 0;
}
Matrix layerNorm(const Matrix& X, const Matrix& gamma, const Matrix& beta, Matrix& Xhat, vector<float>& sigma) {
    int n = X.size();
    int d = X[0].size();
    Matrix res(n, vector<float>(d));
    Xhat = Matrix(n, vector<float>(d));
    sigma = vector<float>(n);

    for (int i = 0;i < n;i++) {
        float sum = 0.0f;
        for (int j = 0;j < d;j++) {
            sum += X[i][j];
        }
        float mean = sum / d;

        sum = 0.0f;
        for (int j = 0;j < d;j++) {
            sum += (X[i][j] - mean) * (X[i][j] - mean);
        }
        float var = sum / d;

        sigma[i] = sqrt(var + 1e-5);

        for (int j = 0;j < d;j++) {
            Xhat[i][j] = (X[i][j] - mean) / sigma[i];
            res[i][j] = gamma[0][j] * Xhat[i][j] + beta[0][j];
        }
    }

    return res;
}
Matrix FNN(const Matrix& X, const Matrix& Wup, const Matrix& Wdown, Matrix& Hact) {
    Matrix H = matmul(X, Wup);
    Hact = Matrix(H.size(), vector<float>(H[0].size()));
    for (int i = 0;i < H.size();i++) {
        for (int j = 0;j < H[0].size();j++) {
            Hact[i][j] = ReLU(H[i][j]);
        }
    }
    Matrix res = matmul(Hact, Wdown);
    return res;
}
/*      Backward Function       */
void softmax_crossentropy_backward(const Matrix& X, const vector<int>& window, Matrix& dLogits) {
    Matrix res(X.size(), vector<float>(X[0].size(), 0.0f));
    Matrix sftmx = softmax(X);
    int n = sftmx.size();
    int cnt = n - lossFrom;
    if (cnt <= 0) {
        dLogits = res;
        return;
    }
    float inv = 1.0f / cnt;
    for (int i = 0;i < n - 1;i++) {
        if (i + 1 < lossFrom)continue;
        int ans = window[i + 1];
        res[i] = sftmx[i];
        res[i][ans] -= 1.0f;
        for (int j = 0;j < (int)res[i].size();j++) {
            res[i][j] *= inv;
        }
    }
    dLogits = res;
}
void matmul_backward(const Matrix& A, const Matrix& B, const Matrix& dC, Matrix& dA, Matrix& dB) {
    dA = matmul(dC, transpose(B));
    dB = matmul(transpose(A), dC);
}
void softmax_attention_backward(const Matrix& scores, const Matrix& grad_scores, Matrix& sftmx) {
    Matrix res(scores.size(), vector<float>(scores[0].size()));
    for (int i = 0;i < scores.size();i++) {
        float sum = 0.0f;
        for (int j = 0;j < scores[0].size();j++) {
            sum += grad_scores[i][j] * scores[i][j];
        }
        for (int j = 0;j < scores[0].size();j++) {
            res[i][j] = scores[i][j] * (grad_scores[i][j] - sum);
        }
    }
    sftmx = res;
}
void applyCausalMask_backward(const Matrix& grad_scores, Matrix& dCM) {
    int n = grad_scores.size();
    Matrix res(n, vector<float>(n, 0.0f));

    for (int i = 0;i < n;i++) {
        for (int j = 0;j <= i;j++) {
            res[i][j] = grad_scores[i][j];
        }
    }
    dCM = res;
}
void scale_backward(const Matrix& dCM, Matrix& dS) {
    Matrix res(dCM.size(), vector<float>(dCM[0].size()));
    for (int i = 0;i < dCM.size();i++) {
        for (int j = 0;j < dCM[0].size();j++) {
            res[i][j] = dCM[i][j] / sqrt(dk / h);
        }
    }
    dS = res;
}

void embedLookup_backward(const vector<int>& window, const Matrix& tokenVecs, Matrix& dwindow) {
    int d = tokenVecs[0].size();
    Matrix res(vocab_size, vector<float>(d, 0.0f));

    for (int i = 0;i < window.size();i++) {
        int id = window[i];
        for (int j = 0;j < d;j++) {
            res[id][j] += tokenVecs[i][j];
        }
    }
    dwindow = res;
}
void update(Matrix& W, const Matrix& grad, float lr) {
    for (int i = 0;i < W.size();i++) {
        for (int j = 0;j < W[0].size();j++) {
            W[i][j] = W[i][j] - lr * grad[i][j];
        }
    }
}
void update_partial(Matrix& W, const Matrix& grad, float lr) {
    for (int i = 0;i < grad.size();i++) {
        for (int j = 0;j < grad[0].size();j++) {
            W[i][j] = W[i][j] - lr * grad[i][j];
        }
    }
}
void adamUpdate(Matrix& W, const Matrix& grad, AdamState& s, int t, float lr) {
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    if (s.m.empty()) {
        s.m = Matrix(W.size(), vector<float>(W[0].size(), 0.0f));
        s.v = Matrix(W.size(), vector<float>(W[0].size(), 0.0f));
    }
    for (int i = 0;i < (int)W.size();i++) {
        for (int j = 0;j < (int)W[0].size();j++) {
            s.m[i][j] = beta1 * s.m[i][j] + (1 - beta1) * grad[i][j];
            s.v[i][j] = beta2 * s.v[i][j] + (1 - beta2) * grad[i][j] * grad[i][j];
            float mhat = s.m[i][j] / (1 - pow(beta1, t));
            float vhat = s.v[i][j] / (1 - pow(beta2, t));
            W[i][j] -= lr * mhat / (sqrt(vhat) + eps);
        }
    }
}
void adamUpdate_partial(Matrix& W, const Matrix& grad, AdamState& s, int t, float lr) {
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    if (s.m.empty()) {
        s.m = Matrix(W.size(), vector<float>(W[0].size(), 0.0f));
        s.v = Matrix(W.size(), vector<float>(W[0].size(), 0.0f));
    }
    for (int i = 0;i < (int)grad.size();i++) {
        for (int j = 0;j < (int)grad[0].size();j++) {
            s.m[i][j] = beta1 * s.m[i][j] + (1 - beta1) * grad[i][j];
            s.v[i][j] = beta2 * s.v[i][j] + (1 - beta2) * grad[i][j] * grad[i][j];
            float mhat = s.m[i][j] / (1 - pow(beta1, t));
            float vhat = s.v[i][j] / (1 - pow(beta2, t));
            W[i][j] -= lr * mhat / (sqrt(vhat) + eps);
        }
    }
}
vector<int> window_select(const vector<int>& tokenIds, mt19937& gen) {
    int total = tokenIds.size();
    int winLen = min(max_seq_len, total);
    int maxStart = total - winLen;
    uniform_int_distribution<int> dist(0, maxStart);
    int start = dist(gen);
    return vector<int>(tokenIds.begin() + start, tokenIds.begin() + start + winLen);
}
void clip(Matrix& grad, float limit) {
    for (int i = 0;i < (int)grad.size();i++) {
        for (int j = 0;j < (int)grad[0].size();j++) {
            if (grad[i][j] > limit) grad[i][j] = limit;
            if (grad[i][j] < -limit) grad[i][j] = -limit;
        }
    }
}
void save(BlockParams& p, const Matrix& Wout, const Matrix& tokenEmbedding, const Matrix& posEmbedding, int l) {
    ofstream tokenEmbedder("train/tokenEmbedding.txt");
    ofstream posEmbedder("train/posEmbedding.txt");
    ofstream WQr("train/WQ_" + to_string(l) + ".txt");
    ofstream WKr("train/WK_" + to_string(l) + ".txt");
    ofstream WVr("train/WV_" + to_string(l) + ".txt");
    ofstream Wor("train/Wo_" + to_string(l) + ".txt");
    ofstream Wouter("train/Wout.txt");
    ofstream gammar("train/gamma_" + to_string(l) + ".txt");
    ofstream betar("train/beta_" + to_string(l) + ".txt");
    ofstream gammar2("train/gamma2_" + to_string(l) + ".txt");
    ofstream betar2("train/beta2_" + to_string(l) + ".txt");
    ofstream Wupr("train/Wup_" + to_string(l) + ".txt");
    ofstream Wdownr("train/Wdown_" + to_string(l) + ".txt");

    tokenEmbedder << vocab_size << ' ' << d << endl;
    for (int i = 0;i < vocab_size;i++) {
        for (int j = 0;j < d;j++) {
            tokenEmbedder << tokenEmbedding[i][j] << ' ';
        }
        tokenEmbedder << endl;
    }
    tokenEmbedder.close();

    posEmbedder << max_seq_len << ' ' << d << endl;
    for (int i = 0;i < max_seq_len;i++) {
        for (int j = 0;j < d;j++) {
            posEmbedder << posEmbedding[i][j] << ' ';
        }
        posEmbedder << endl;
    }
    posEmbedder.close();

    WQr << d << ' ' << dk << endl;
    WKr << d << ' ' << dk << endl;
    WVr << d << ' ' << dk << endl;
    Wor << d << ' ' << dk << endl;
    for (int i = 0;i < d;i++) {
        for (int j = 0;j < dk;j++) {
            WQr << p.WQ[i][j] << ' ';
            WKr << p.WK[i][j] << ' ';
            WVr << p.WV[i][j] << ' ';
            Wor << p.Wo[i][j] << ' ';
        }
        WQr << endl;
        WKr << endl;
        WVr << endl;
        Wor << endl;
    }
    WQr.close();
    WKr.close();
    WVr.close();
    Wor.close();

    Wouter << d << ' ' << vocab_size << endl;
    for (int i = 0;i < d;i++) {
        for (int j = 0;j < vocab_size;j++) {
            Wouter << Wout[i][j] << ' ';
        }
        Wouter << endl;
    }
    Wouter.close();


    gammar << 1 << ' ' << d << endl;
    betar << 1 << ' ' << d << endl;
    for (int i = 0;i < d;i++) {
        gammar << p.gamma[0][i] << ' ';
        betar << p.beta[0][i] << ' ';
    }
    gammar.close();
    betar.close();

    gammar2 << 1 << ' ' << d << endl;
    betar2 << 1 << ' ' << d << endl;
    for (int i = 0;i < d;i++) {
        gammar2 << p.gamma2[0][i] << ' ';
        betar2 << p.beta2[0][i] << ' ';
    }
    gammar2.close();
    betar2.close();

    Wupr << d << ' ' << dimension << endl;
    Wdownr << dimension << ' ' << d << endl;
    for (int i = 0;i < d;i++) {
        for (int j = 0;j < dimension;j++) {
            Wupr << p.Wup[i][j] << ' ';
        }
        Wupr << endl;
    }
    for (int i = 0;i < dimension;i++) {
        for (int j = 0;j < d;j++) {
            Wdownr << p.Wdown[i][j] << ' ';
        }
        Wdownr << endl;
    }
    Wupr.close();
    Wdownr.close();
}
void save_loss(float loss) {
    ofstream Losser("train/Loss.txt", ios::app);
    Losser << loss << endl;
    Losser.close();
}
void layerNorm_backward(const Matrix& dXnorm, const Matrix& Xhat, const vector<float>& sigma, const Matrix& gamma, Matrix& dXnoXrc, Matrix& dgamma, Matrix& dbeta) {
    int n = dXnorm.size();
    int d = dXnorm[0].size();

    Matrix dXhat(n, vector<float>(d));
    dXnoXrc = Matrix(n, vector<float>(d));
    dgamma = Matrix(1, vector<float>(d, 0.0f));
    dbeta = Matrix(1, vector<float>(d, 0.0f));
    for (int i = 0;i < n;i++) {
        for (int j = 0;j < d;j++) {
            dXhat[i][j] = dXnorm[i][j] * gamma[0][j];
        }

        float sum = 0.0f;
        for (int j = 0;j < d;j++) {
            sum += dXhat[i][j];
        }
        float mean_dXhat = sum / d;

        sum = 0.0f;
        for (int j = 0;j < d;j++) {
            sum += dXhat[i][j] * Xhat[i][j];
        }
        float mean_dXhat_Xhat = sum / d;

        sum = 0.0f;
        for (int j = 0;j < d;j++) {
            dXnoXrc[i][j] = (dXhat[i][j] - mean_dXhat - Xhat[i][j] * mean_dXhat_Xhat) / sigma[i];
            dgamma[0][j] += dXnorm[i][j] * Xhat[i][j];
            dbeta[0][j] += dXnorm[i][j];
        }
    }
}
void FNN_backward(const Matrix& dFNNout, const Matrix& Xnorm2, const Matrix& Hact, const Matrix& Wup, const Matrix& Wdown, Matrix& dXnorm2, Matrix& dWup, Matrix& dWdown) {
    Matrix dHact;
    matmul_backward(Hact, Wdown, dFNNout, dHact, dWdown);

    Matrix dH(dHact.size(), vector<float>(dHact[0].size()));
    for (int i = 0;i < dH.size();i++) {
        for (int j = 0;j < dH[0].size();j++) {
            dH[i][j] = (Hact[i][j] > 0) ? dHact[i][j] : 0;
        }
    }

    matmul_backward(Xnorm2, Wup, dH, dXnorm2, dWup);
}
Matrix block_forward(const Matrix& X, const BlockParams& p, BlockCache& c) {
    /*layerNorm 1*/
    c.Xnorm = layerNorm(X, p.gamma, p.beta, c.Xhat, c.sigma);
    /*Multi-head attention*/
    c.A = multi_head_attention(c.Xnorm, p.WQ, p.WK, p.WV, c.Qcut, c.Kcut, c.Ktcut, c.Vcut, c.scores);
    Matrix AWo = matmul(c.A, p.Wo);
    /*Residual Connection 1*/
    c.Xrc = matrix_add(X, AWo);
    /*layerNorm 2*/
    c.Xnorm2 = layerNorm(c.Xrc, p.gamma2, p.beta2, c.Xhat2, c.sigma2);
    /*FNN Connection*/
    Matrix FNNout = FNN(c.Xnorm2, p.Wup, p.Wdown, c.Hact);
    /*Residual Connection 2*/
    c.Xrc2 = matrix_add(c.Xrc, FNNout);

    return c.Xrc2;
}
Matrix block_backward(const Matrix& grad_dXrc2, const BlockParams& p, const BlockCache& c, BlockParamGrads& g) {
    /*FNN backward*/
    Matrix grad_dFNNout = grad_dXrc2;
    Matrix grad_dXnorm2, grad_dWup, grad_dWdown;
    FNN_backward(grad_dFNNout, c.Xnorm2, c.Hact, p.Wup, p.Wdown, grad_dXnorm2, g.grad_dWup, g.grad_dWdown);

    /*layerNorm 2 backward*/
    Matrix grad_dXnoXrc2;
    layerNorm_backward(grad_dXnorm2, c.Xhat2, c.sigma2, p.gamma2, grad_dXnoXrc2, g.grad_dgamma2, g.grad_dbeta2);
    Matrix grad_dXrc = matrix_add(grad_dXnoXrc2, grad_dXrc2);

    /*Multi-head attention backward*/
    Matrix grad_dAWo = grad_dXrc;
    Matrix grad_dA;
    matmul_backward(c.A, p.Wo, grad_dAWo, grad_dA, g.grad_dWo);

    vector<Matrix> grad_dAcut;
    for (int m = 0;m < h;m++) {
        Matrix grad_dAcut_m(grad_dA.size(), vector<float>((grad_dA[0].size() / h)));
        for (int i = 0;i < grad_dA.size();i++) {
            for (int j = 0;j < (grad_dA[0].size() / h);j++) {
                grad_dAcut_m[i][j] = grad_dA[i][m * (grad_dA[0].size()) / h + j];
            }
        }
        grad_dAcut.push_back(grad_dAcut_m);
    }
    Matrix grad_scores, grad_dV;
    Matrix grad_sftmx_scores;
    Matrix grad_dCM;
    Matrix grad_dS;
    Matrix grad_dQ, grad_dKt;
    matmul_backward(c.scores[0], c.Vcut[0], grad_dAcut[0], grad_scores, grad_dV);
    softmax_attention_backward(c.scores[0], grad_scores, grad_sftmx_scores);
    applyCausalMask_backward(grad_sftmx_scores, grad_dCM);
    scale_backward(grad_dCM, grad_dS);
    matmul_backward(c.Qcut[0], c.Ktcut[0], grad_dS, grad_dQ, grad_dKt);
    Matrix grad_dK = transpose(grad_dKt);
    for (int m = 1;m < h;m++) {
        /*Attention Scores/V backward*/
        Matrix grad_scores_m, grad_dV_m;
        matmul_backward(c.scores[m], c.Vcut[m], grad_dAcut[m], grad_scores_m, grad_dV_m);
        /*Attention Softmax backward*/
        Matrix grad_sftmx_scores_m;
        softmax_attention_backward(c.scores[m], grad_scores_m, grad_sftmx_scores_m);
        /*Attention CM backward*/
        Matrix grad_dCM_m;
        applyCausalMask_backward(grad_sftmx_scores_m, grad_dCM_m);
        /*Attention Scale backward*/
        Matrix grad_dS_m;
        scale_backward(grad_dCM_m, grad_dS_m);
        /*Attention Q/K backward*/
        Matrix grad_dQ_m, grad_dKt_m;
        matmul_backward(c.Qcut[m], c.Ktcut[m], grad_dS_m, grad_dQ_m, grad_dKt_m);
        Matrix grad_dK_m = transpose(grad_dKt_m);

        grad_dV = concat(grad_dV, grad_dV_m);
        grad_dQ = concat(grad_dQ, grad_dQ_m);
        grad_dK = concat(grad_dK, grad_dK_m);
    }
    /*Attention WQKV backward*/
    Matrix grad_dWQ, grad_dWK, grad_dWV, grad_dXQ, grad_dXK, grad_dXV;
    matmul_backward(c.Xnorm, p.WQ, grad_dQ, grad_dXQ, g.grad_dWQ);
    matmul_backward(c.Xnorm, p.WK, grad_dK, grad_dXK, g.grad_dWK);
    matmul_backward(c.Xnorm, p.WV, grad_dV, grad_dXV, g.grad_dWV);
    Matrix grad_dXnorm = matrix_add(matrix_add(grad_dXQ, grad_dXK), grad_dXV);

    /*layerNorm 1 backward*/
    Matrix grad_dXnoXrc, grad_dgamma, grad_dbeta;
    layerNorm_backward(grad_dXnorm, c.Xhat, c.sigma, p.gamma, grad_dXnoXrc, g.grad_dgamma, g.grad_dbeta);

    return matrix_add(grad_dXnoXrc, grad_dXrc);
}
float dynamic_lr(float lr_max, float lr_min, int e) {
    float res = lr_min;
    float minus = lr_max - lr_min;
    float d = 0.5 * (1 + cos(e * M_PI / epoch));
    return res + minus * d;
}
void read_performance(int& e_start) {
    ifstream file1("train/para.txt");
    ifstream file2("train/Loss.txt");
    string str;
    if (file1) {
        getline(file1, str);
        epoch = stoi(str);
        getline(file1, str);
        d = stoi(str);
        getline(file1, str);
        dk = stoi(str);
        getline(file1, str);
        h = stoi(str);
        getline(file1, str);
        N = stoi(str);
        getline(file1, str);
        max_seq_len = stoi(str);
    }

    if (file2) {
        int i = 0;
        while (getline(file2, str)) {
            i++;
        }
        e_start = i * record;
    }
}
int main()
{
    initCUDA();
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
    _chdir("F:\\Transformer_Sweden_CUDA\\TSFM_USR_Sweden\\x64\\Release");

    random_device rd;
    mt19937 gen(rd());

    vector<BlockParams> layers(N);
    vector<BlockParamGrads> grads(N);
    /*      Loading data     */
    BPE_load("train/bpe_vocab.txt", "train/bpe_merges.txt");

    Matrix tokenEmbedding = Matrix_load("train/tokenEmbedding.txt");
    Matrix posEmbedding = Matrix_load("train/posEmbedding.txt");
    Matrix Wout = Matrix_load("train/Wout.txt");
    for (int l = 0;l < N;l++) {
        layers[l].WQ = Matrix_load("train/WQ_" + to_string(l) + ".txt");
        layers[l].WK = Matrix_load("train/WK_" + to_string(l) + ".txt");
        layers[l].WV = Matrix_load("train/WV_" + to_string(l) + ".txt");
        layers[l].Wo = Matrix_load("train/Wo_" + to_string(l) + ".txt");
        layers[l].gamma = Matrix_load("train/gamma_" + to_string(l) + ".txt");
        layers[l].beta = Matrix_load("train/beta_" + to_string(l) + ".txt");
        layers[l].Wup = Matrix_load("train/Wup_" + to_string(l) + ".txt");
        layers[l].Wdown = Matrix_load("train/Wdown_" + to_string(l) + ".txt");
        layers[l].gamma2 = Matrix_load("train/gamma2_" + to_string(l) + ".txt");
        layers[l].beta2 = Matrix_load("train/beta2_" + to_string(l) + ".txt");
    }


    vector<vector<int>> sftIds;
    vector<int> sftFrom;
    {
        ifstream f("train/sft.txt");
        int assistant = token2id["<|assistant|>"];
        string line;
        while (getline(f, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                continue;
            }
            vector<int> ids = tokenize(line);
            if ((int)ids.size() > max_seq_len || (int)ids.size() < 3)continue;
            int b = -1;
            for (int i = 0;i < (int)ids.size();i++) {
                if (ids[i] == assistant) {
                    b = i;
                    break;
                }
            }
            if (b < 0 || b + 1 >= (int)ids.size()) {
                continue;
            }
            sftIds.push_back(ids);
            sftFrom.push_back(b + 1);
        }
        cout << "SFT samples: " << sftIds.size() << endl;
    }
    uniform_int_distribution<int> pick(0, (int)sftIds.size() - 1);

    int r = 1;
    float average = 0;
    /*Adam*/
    vector<BlockAdamState> adamStates(N);
    AdamState adamWout, adamTokenEmb, adamPosEmb;
    int t = 0;

    cout << "Storage(MB): " << float((N * (12 * d * d + 4 * d) + 2 * vocab_size * d + max_seq_len * d)) / 4.19e6 << "MB" << endl;
    int e_start;
    read_performance(e_start);
    for (int e = e_start;e < epoch;e++) {
        vector<BlockCache> caches(N);
        /*      Forward     */
        int si = pick(gen);
        vector<int> window = sftIds[si];
        lossFrom = sftFrom[si];

        Matrix posSlice(window.size(), vector<float>(d));
        for (int i = 0;i < (int)window.size();i++)posSlice[i] = posEmbedding[i];
        Matrix X = matrix_add(embedLookup(window, tokenEmbedding), posSlice);

        Matrix layer_input = X;
        for (int l = 0;l < N;l++) {
            layer_input = block_forward(layer_input, layers[l], caches[l]);
        }

        Matrix result = matmul(layer_input, Wout);

        float loss = losslize(result, window);

        /*      Backward        */

        /*Softmax backward*/
        Matrix grad_dC;
        softmax_crossentropy_backward(result, window, grad_dC);

        /*result / Residual connection/AWo backward*/
        Matrix grad_dXrc2, grad_dWout;
        matmul_backward(caches[N - 1].Xrc2, Wout, grad_dC, grad_dXrc2, grad_dWout);

        Matrix grad_dLayerOut = grad_dXrc2;

        for (int l = N - 1;l >= 0;l--) {
            grad_dLayerOut = block_backward(grad_dLayerOut, layers[l], caches[l], grads[l]);
        }

        /*Encoding backward*/
        Matrix grad_tokenVecs = grad_dLayerOut;
        Matrix grad_posTable_partial = grad_dLayerOut;
        /*Embedding backward*/
        Matrix grad_dwindow;
        embedLookup_backward(window, grad_tokenVecs, grad_dwindow);

        /*      Machine Learning        */
        t++;
        /*Test clip range*/
        float maxg = 0;
        for (auto& row : grads[0].grad_dWQ) for (auto v : row) maxg = max(maxg, fabs(v));
        float lr = dynamic_lr(2e-5f, 2e-6f, e);
        float clipLimit = 1.0f;

        clip(grad_dWout, clipLimit);
        clip(grad_dwindow, clipLimit);
        clip(grad_posTable_partial, clipLimit);
        adamUpdate(Wout, grad_dWout, adamWout, t, lr);
        adamUpdate(tokenEmbedding, grad_dwindow, adamTokenEmb, t, lr);
        adamUpdate_partial(posEmbedding, grad_posTable_partial, adamPosEmb, t, lr);
        for (int l = 0;l < N;l++) {
            clip(grads[l].grad_dWQ, clipLimit);
            clip(grads[l].grad_dWK, clipLimit);
            clip(grads[l].grad_dWV, clipLimit);
            clip(grads[l].grad_dWo, clipLimit);
            clip(grads[l].grad_dgamma, clipLimit);
            clip(grads[l].grad_dbeta, clipLimit);
            clip(grads[l].grad_dgamma2, clipLimit);
            clip(grads[l].grad_dbeta2, clipLimit);
            clip(grads[l].grad_dWup, clipLimit);
            clip(grads[l].grad_dWdown, clipLimit);

            adamUpdate(layers[l].WQ, grads[l].grad_dWQ, adamStates[l].WQ, t, lr);
            adamUpdate(layers[l].WK, grads[l].grad_dWK, adamStates[l].WK, t, lr);
            adamUpdate(layers[l].WV, grads[l].grad_dWV, adamStates[l].WV, t, lr);
            adamUpdate(layers[l].Wo, grads[l].grad_dWo, adamStates[l].Wo, t, lr);
            adamUpdate(layers[l].gamma, grads[l].grad_dgamma, adamStates[l].gamma, t, lr);
            adamUpdate(layers[l].beta, grads[l].grad_dbeta, adamStates[l].beta, t, lr);
            adamUpdate(layers[l].gamma2, grads[l].grad_dgamma2, adamStates[l].gamma2, t, lr);
            adamUpdate(layers[l].beta2, grads[l].grad_dbeta2, adamStates[l].beta2, t, lr);
            adamUpdate(layers[l].Wup, grads[l].grad_dWup, adamStates[l].Wup, t, lr);
            adamUpdate(layers[l].Wdown, grads[l].grad_dWdown, adamStates[l].Wdown, t, lr);
        }
        cout << "LOSS:" << loss << "][Epoch:" << e << "/" << epoch << "][grad_dWQ max abs:" << maxg << endl;
        average += loss;
        if (r == record) {
            for (int l = 0;l < N;l++) {
                save(layers[l], Wout, tokenEmbedding, posEmbedding, l);
            }
            save_loss(average / (float)record);
            r = 0;
            average = 0;
        }
        r++;
    }
    return 0;
}