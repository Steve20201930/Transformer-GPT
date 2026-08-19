#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <algorithm>
#include <random>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map> 
#include <filesystem>
#include <direct.h>
#include <windows.h>
#include <cuda_runtime.h>
using namespace std;

typedef vector<vector<double>> Matrix;
int vocab_size = 0, d = 384, dk = 384;
int h = 6; //The Multi-head quantity must be checked
int N = 8;
int max_seq_len = 256;
int max_length = 500;
double temp = 0.35f;
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
    vector<double> sigma;
    Matrix Xnorm;
    vector<Matrix> Qcut, Kcut, Ktcut, Vcut, scores;
    Matrix A, Xrc;
    Matrix Xhat2;
    vector<double> sigma2;
    Matrix Xnorm2, Hact, Xrc2;
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
__global__ void matmul_kernel(const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    int M, int K, int N, int tA, int tB)
{
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;   // 对应原来的 gid.y
    int col = blockIdx.x * TILE + threadIdx.x;   // 对应原来的 gid.x

    double sum = 0.0f;
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
    double* dev = nullptr;
    double* host = nullptr;   // pinned，用于 H2D / D2H 暂存
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

    size_t szA = (size_t)aR * aC * sizeof(double);
    size_t szB = (size_t)bR * bC * sizeof(double);
    size_t szC = (size_t)M * N * sizeof(double);

    ensureCap(bufA, szA);
    ensureCap(bufB, szB);
    ensureCap(bufC, szC);

    // 按物理形状原样打平，转置交给 kernel
    for (int i = 0; i < aR; i++)
        memcpy(bufA.host + (size_t)i * aC, A[i].data(), aC * sizeof(double));
    for (int i = 0; i < bR; i++)
        memcpy(bufB.host + (size_t)i * bC, B[i].data(), bC * sizeof(double));

    CUDA_CHECK(cudaMemcpy(bufA.dev, bufA.host, szA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bufB.dev, bufB.host, szB, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_kernel << <grid, block >> > (bufA.dev, bufB.dev, bufC.dev,
        M, K, N, tA ? 1 : 0, tB ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(bufC.host, bufC.dev, szC, cudaMemcpyDeviceToHost));

    Matrix res(M, vector<double>(N));
    for (int i = 0; i < M; i++)
        memcpy(res[i].data(), bufC.host + (size_t)i * N, N * sizeof(double));
    return res;
}

/* ================== CUDA 结束 ================== */

Matrix transpose(const Matrix& A) {
    int Arow = A.size();
    int Acolumn = A[0].size();
    Matrix res(Acolumn, vector<double>(Arow, 0.0f));

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
    Matrix res(Arow, vector<double>(Acolumn, 0.0f));

    for (int i = 0;i < Arow;i++) {
        double e_sum = 0.0f;
        double maxv = A[i][0];
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
    Matrix res(n, vector<double>(n, -1e9f));

    for (int i = 0;i < n;i++) {
        for (int j = 0;j <= i;j++) {
            res[i][j] = CM[i][j];
        }
    }
    return res;
}
Matrix attention(const Matrix& X, const Matrix& WQ, const Matrix& WK, const Matrix& WV, Matrix& Q, Matrix& K, Matrix& Kt, Matrix& V, Matrix& scores) {
    /*
        Attention = Softmax((Q x Kt)/sqrt(dk)) x V
    */
    int n = X.size();
    int d = X[0].size();
    int dk = WQ[0].size();

    Q = matmul(X, WQ); // X x WQ
    K = matmul(X, WK); // X x WK
    V = matmul(X, WV); // X x WV
    Kt = transpose(K); // Kt

    scores = matmul(Q, Kt);

    //Divided by sqrt(dk) and softmax
    for (int i = 0;i < n;i++) {
        for (int j = 0;j < n;j++) {
            scores[i][j] /= sqrt(dk);
        }
    }
    scores = applyCausalMask(scores);
    scores = softmax(scores);

    Matrix res = matmul(scores, V);
    return res;
}
Matrix concat(const Matrix& A, const Matrix& B) {
    Matrix res(A.size(), vector<double>(A[0].size() + B[0].size()));
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
    Matrix scoresConcat(X.size(), vector<double>(0));
    Matrix head_out;
    Matrix res(X.size(), vector<double>(0));

    for (int m = 0;m < h;m++) {
        Matrix Qcut_m(X.size(), vector<double>(dk / h));
        Matrix Kcut_m(X.size(), vector<double>(dk / h));
        Matrix Vcut_m(X.size(), vector<double>(dk / h));
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

        head_out = matmul(scoresCut, Vcut_m);

        res = concat(res, head_out);
    }
    return res;
}
Matrix Matrix_load(string filename) {

    ifstream file(filename);
    int a, b;
    file >> a >> b;

    Matrix res(a, vector<double>(b));
    for (int i = 0;i < a;i++) {
        for (int j = 0;j < b;j++) {
            file >> res[i][j];
        }
    }
    return res;
}
double ReLU(double x) {
    return (x >= 0) ? x : 0;
}
Matrix layerNorm(const Matrix& X, const Matrix& gamma, const Matrix& beta, Matrix& Xhat, vector<double>& sigma) {
    int n = X.size();
    int d = X[0].size();
    Matrix res(n, vector<double>(d));
    Xhat = Matrix(n, vector<double>(d));
    sigma = vector<double>(n);

    for (int i = 0;i < n;i++) {
        double sum = 0.0f;
        for (int j = 0;j < d;j++) {
            sum += X[i][j];
        }
        double mean = sum / d;

        sum = 0.0f;
        for (int j = 0;j < d;j++) {
            sum += (X[i][j] - mean) * (X[i][j] - mean);
        }
        double var = sum / d;

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
    Hact = Matrix(H.size(), vector<double>(H[0].size()));
    for (int i = 0;i < H.size();i++) {
        for (int j = 0;j < H[0].size();j++) {
            Hact[i][j] = ReLU(H[i][j]);
        }
    }
    Matrix res = matmul(Hact, Wdown);
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
    ifstream f(vocabFile, ios::binary);  // 用二进制模式，避免换行符转换干扰

    // 跳过 UTF-8 BOM（如果有的话）
    unsigned char bom[3];
    f.read((char*)bom, 3);
    if (bom[0] != 0xEF || bom[1] != 0xBB || bom[2] != 0xBF) {
        f.seekg(0);  // 不是 BOM，退回去
    }

    int size;
    f >> size;
    f.ignore(256, '\n');  // ✅ 跳过整行剩余内容（包括 \r\n）

    for (int i = 0; i < size; i++) {
        string line;
        getline(f, line);

        // ✅ 去掉末尾的 \r
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        size_t sp = line.find(' ');
        string token = line.substr(sp + 1);

        id2token.push_back(token);
        token2id[token] = i;
    }
    f.close();

    ifstream g(mergeFile, ios::binary);  // ✅ 也用二进制模式
    int m;
    g >> m;
    for (int i = 0; i < m; i++) {
        int a, b, nid;
        g >> a >> b >> nid;
        pairRank[{a, b}] = i;
        rankPair.push_back({ a, b });
        rankNew.push_back(nid);
    }
    g.close();

    vocab_size = id2token.size();
    UNK = token2id["<|unk|>"];
    for (int i = 0; i < (int)id2token.size(); i++) {
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
    Matrix res(window.size(), vector<double>(tokenTable[0].size()));

    for (int i = 0;i < window.size();i++) {
        res[i] = tokenTable[window[i]];
    }
    return res;
}
Matrix matrix_add(const Matrix& A, const Matrix& B) {
    Matrix res(A.size(), vector<double>(A[0].size()));
    for (int i = 0;i < A.size();i++) {
        for (int j = 0;j < A[0].size();j++) {
            res[i][j] = A[i][j] + B[i][j];
        }
    }
    return res;
}
int find_next(const Matrix& result) {
    int res = 0;
    double max = result[result.size() - 1][0];
    for (int i = 1;i < result[0].size();i++) {
        if (result[result.size() - 1][i] > max) {
            res = i;
            max = result[result.size() - 1][i];
        }
    }
    return res;
}
int sample_next(const Matrix& result, mt19937& gen, const vector<int>& full, int from) {
    vector<double> logits = result.back();
    int V = logits.size();

    for (int i = from;i < (int)full.size();i++) {
        logits[full[i]] -= 4.0f;
    }

    for (int i = 0;i < V;i++) {
        logits[i] /= temp;
    }
    //Softmax
    double mx = logits[0];
    for (int i = 1;i < V;i++) {
        mx = max(mx, logits[i]);
    }
    double sum = 0;
    vector<double> p(V);
    for (int i = 0;i < V;i++) {
        p[i] = exp(logits[i] - mx);
        sum += p[i];
    }
    for (int i = 0;i < V;i++) {
        p[i] /= sum;
    }
    //Random select
    uniform_real_distribution<double> dis(0.0f, 1.0f);
    double r = dis(gen), acc = 0;
    for (int i = 0;i < V;i++) {
        acc += p[i];
        if (r <= acc) return i;
    }
    return V - 1;
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
void read_performance() {
    ifstream file1("train/para.txt");
    string str;
    if (file1) {
        getline(file1, str);
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
}
/*Users zone*/
string front = "<|user|>";
string opposite = "<|assistant|>";
string clipHistory(const string& hist, string& usrRef) {
    string cur = front + usrRef + opposite;
    int budget = max_seq_len - 120;
    while ((int)tokenize(cur).size() > budget && !usrRef.empty()) {
        size_t n = usrRef.size();
        size_t cut = n > 30 ? n - 30 : 0;
        while (cut < n && ((unsigned char)usrRef[cut] & 0xC0) == 0x80) cut++;
        usrRef = usrRef.substr(0, cut);
        cur = front + usrRef + opposite;
    }
    string h = hist;
    while (!h.empty() && (int)tokenize(h + cur).size() > budget) {
        size_t p = h.find("<|end|>");
        if (p == string::npos) {
            h.clear();
            break;
        }
        h = h.substr(p + 7);
    }
    return h;
}

int main()
{
    initCUDA();
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
    _chdir("F:\\Transformer_Sweden_CUDA\\TSFM_USR_Sweden\\x64\\Release");
    read_performance();
    bool debug = false;
    bool mem = false;
    cout << "front: " << front << endl;
    cout << "opposite: " << opposite << endl;
    cout << "temperature: " << temp << endl;
    cout << "max length: " << max_length << endl;

    mt19937 gen(random_device{}());
    /*      Loading data     */
    BPE_load("train/bpe_vocab.txt", "train/bpe_merges.txt");

    Matrix tokenEmbedding = Matrix_load("train/tokenEmbedding.txt");
    Matrix posEmbedding = Matrix_load("train/posEmbedding.txt");
    Matrix Wout = Matrix_load("train/Wout.txt");
    vector<BlockParams> layers(N);
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

    string history = "";
    while (true) {
        cout << "> ";
        string usr;
        if (!getline(cin, usr))break;
        if (usr.empty())continue;
        if (usr == "#change") {
            string str;
            cout << "front: ";
            getline(cin, str);
            front = str;
            cout << "opposite: ";
            getline(cin, str);
            opposite = str;
            continue;
        }
        if (usr == "#temp") {
            string str;
            cout << "temp: ";
            getline(cin, str);
            temp = stof(str);
            continue;
        }
        if (usr == "#maxlength") {
            string str;
            cout << "Max length: ";
            getline(cin, str);
            max_length = stoi(str);
            continue;
        }
        if (usr == "#debug") {
            debug = !debug;
            continue;
        }
        if (usr == "#mem") {
            mem = !mem;
            continue;
        }

        history = clipHistory(history, usr);
        string input = history + front + usr + opposite;
        if (mem)history += front + usr + opposite;

        int END = token2id.count("<|end|>") ? token2id["<|end|>"] : -1;

        vector<int> full = tokenize(input);
        int promptLen = (int)full.size();

        for (int m = 0;m < max_length;m++) {
            /*      Forward     */
            vector<BlockCache> caches(N);
            vector<int> window;
            int start = max(0, (int)full.size() - max_seq_len);
            for (int i = start;i < (int)full.size();i++) {
                window.push_back(full[i]);
            }

            Matrix posSlice(window.size(), vector<double>(d));
            for (int i = 0;i < (int)window.size();i++) {
                posSlice[i] = posEmbedding[i];
            }
            Matrix X = matrix_add(embedLookup(window, tokenEmbedding), posSlice);

            Matrix layer_input = X;
            for (int l = 0;l < N;l++) {
                layer_input = block_forward(layer_input, layers[l], caches[l]);
            }

            Matrix result = matmul(layer_input, Wout);

            if (m == 0 && debug) {
                vector<double> lg = result.back();
                int V = lg.size();
                vector<int> idx(V);
                for (int i = 0;i < V;i++) idx[i] = i;
                partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                    [&](int a, int b) { return lg[a] > lg[b]; });
                double mx = lg[idx[0]], s = 0.0f;
                for (int i = 0;i < V;i++) s += exp(lg[i] - mx);
                printf("--- step0 top5 (prompt tokens=%d) ---\n", (int)window.size());
                for (int k = 0;k < 5;k++)
                    printf("  %-14s p=%.4f\n", id2token[idx[k]].c_str(),
                        exp(lg[idx[k]] - mx) / s);
                printf("--- last 3 prompt tokens: ");
                for (int i = max(0, (int)window.size() - 3); i < (int)window.size(); i++)
                    printf("[%s] ", id2token[window[i]].c_str());
                printf("---\n");
            }


            int next = sample_next(result, gen, full, promptLen);
            if (next == END)break;

            if (mem)history += id2token[next];
            cout << id2token[next];
            cout.flush();
            input += id2token[next];
            full.push_back(next);
        }
        if (mem)history += "<|end|>";
        cout << endl;
    }
    return 0;
}
