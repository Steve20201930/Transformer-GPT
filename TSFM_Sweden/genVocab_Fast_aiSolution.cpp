#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <string>
#include <cstdint>
using namespace std;

const int TARGET_VOCAB = 16000;
const size_t SAMPLE_BYTES = 20000000;   // 只在 20MB 采样上训练 BPE
const int MIN_FREQ = 5;

/* ---------- 与推理端保持一致的切分 ---------- */
bool isBoundary(const string &x){
    static const set<string> b = {
        "。","，","、","；","：","？","！","…","—",
        "“","”","‘","’","（","）","《","》",
        "\n","\r"," ","　",
        ".",",",";",":","?","!",
        "\"","'","-","(",")","[","]","/","·","~"   // 补 ASCII 标点
    };
    return b.count(x) > 0;
}
void split(const string &x, vector<vector<string>> &corpus){
    int len = 0;
    vector<string> cur;
    for(size_t i = 0;i < x.size();i += len){
        unsigned char t = x[i];
        if(t < 128) len = 1;
        else if(t < 224) len = 2;
        else if(t < 240) len = 3;
        else len = 4;
        string byte = x.substr(i, len);
        if(!isBoundary(byte)){
            cur.push_back(byte);
        }else{
            if(!cur.empty()){ corpus.push_back(cur); cur.clear(); }
            if(byte != " " && byte != "\n" && byte != "\r" && byte != "　")
                corpus.push_back({byte});
        }
    }
    if(!cur.empty()) corpus.push_back(cur);
}

/* ---------- pair 打包成 int64 ---------- */
static inline uint64_t pk(int a, int b){
    return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
}
static inline int pkA(uint64_t k){ return (int)(k >> 32); }
static inline int pkB(uint64_t k){ return (int)(k & 0xffffffffu); }

int main(){
    /* ---------- 读取 + 采样 ---------- */
    ifstream fin("train/train.txt");
    stringstream buf; buf << fin.rdbuf();
    string text = buf.str();
    cout << "raw: " << text.size()/1e6 << " MB" << endl;

    if(text.size() > SAMPLE_BYTES){
        text.resize(SAMPLE_BYTES);
        // 回退到最后一个换行，避免切断 UTF-8
        size_t cut = text.rfind('\n');
        if(cut != string::npos) text.resize(cut);
        cout << "sampled to: " << text.size()/1e6 << " MB" << endl;
    }

    /* ---------- 切分 + 建初始词表 ---------- */
    vector<vector<string>> segsStr;
    split(text, segsStr);
    { string().swap(text); }   // 释放原始文本

    vector<string> id2token;
    unordered_map<string,int> token2id;
    for(const string &s : {"<|unk|>","<|user|>","<|assistant|>","<|end|>"}){
        token2id[s] = (int)id2token.size();
        id2token.push_back(s);
    }

    /* 摊平成一个大数组 + 双向链表
       tok[i] = token id, -1 表示已被吞并
       prv[i]/nxt[i] = 前驱/后继下标, -1 表示 segment 边界   */
    vector<int> tok, prv, nxt;
    tok.reserve(20000000);
    for(size_t s = 0; s < segsStr.size(); s++){
        int base = (int)tok.size();
        int n = (int)segsStr[s].size();
        for(int j = 0; j < n; j++){
            const string &ch = segsStr[s][j];
            auto it = token2id.find(ch);
            int id;
            if(it == token2id.end()){
                id = (int)id2token.size();
                token2id[ch] = id;
                id2token.push_back(ch);
            } else id = it->second;
            tok.push_back(id);
            prv.push_back(j == 0     ? -1 : base + j - 1);
            nxt.push_back(j == n - 1 ? -1 : base + j + 1);
        }
    }
    { vector<vector<string>>().swap(segsStr); }

    long long total = (long long)tok.size();
    cout << "base vocab: " << id2token.size() << endl;
    cout << "total tokens: " << total << endl;

    /* ---------- 初始 pair 统计 + 倒排索引 ---------- */
    unordered_map<uint64_t,int> freq;
    unordered_map<uint64_t, vector<int>> occ;   // pair -> 左端下标列表
    freq.reserve(1 << 22);
    occ.reserve(1 << 22);
    for(int i = 0; i < (int)tok.size(); i++){
        int j = nxt[i];
        if(j < 0) continue;
        uint64_t k = pk(tok[i], tok[j]);
        freq[k]++;
        occ[k].push_back(i);
    }
    cout << "distinct pairs: " << freq.size() << endl;

    /* 惰性删除的最大堆：取出后与当前 freq 核对，不符就丢弃 */
    priority_queue<pair<int,uint64_t>> pq;
    for(auto &e : freq) pq.push({e.second, e.first});

    vector<pair<pair<int,int>,int>> merges;

    /* ---------- 主循环 ---------- */
    while((int)id2token.size() < TARGET_VOCAB){
        uint64_t best = 0; int bestCount = 0;
        while(!pq.empty()){
            auto top = pq.top();
            auto it = freq.find(top.second);
            if(it != freq.end() && it->second == top.first){   // 计数仍然有效
                best = top.second; bestCount = top.first;
                pq.pop();
                break;
            }
            pq.pop();                                          // 过期条目，丢弃
        }
        if(bestCount < MIN_FREQ) break;

        int a = pkA(best), b = pkB(best);
        int newId = (int)id2token.size();
        id2token.push_back(id2token[a] + id2token[b]);
        merges.push_back({{a,b}, newId});

        /* 只遍历这个 pair 出现过的位置 */
        vector<int> sites;
        {
            auto it = occ.find(best);
            if(it != occ.end()){ sites = it->second; occ.erase(it); }
        }
        freq.erase(best);

        unordered_set<uint64_t> touched;

        for(int i : sites){
            if(tok[i] != a) continue;              // 已被上一次 merge 改动
            int j = nxt[i];
            if(j < 0 || tok[j] != b) continue;

            int p = prv[i], q = nxt[j];

            /* 撤销受影响的旧 pair */
            if(p >= 0){
                uint64_t k = pk(tok[p], a);
                if(--freq[k] <= 0) freq.erase(k); else touched.insert(k);
            }
            if(q >= 0){
                uint64_t k = pk(b, tok[q]);
                if(--freq[k] <= 0) freq.erase(k); else touched.insert(k);
            }

            /* 执行合并：i 变成 newId，j 作废 */
            tok[i] = newId;
            tok[j] = -1;
            nxt[i] = q;
            if(q >= 0) prv[q] = i;

            /* 登记新产生的 pair */
            if(p >= 0){
                uint64_t k = pk(tok[p], newId);
                freq[k]++; occ[k].push_back(p); touched.insert(k);
            }
            if(q >= 0){
                uint64_t k = pk(newId, tok[q]);
                freq[k]++; occ[k].push_back(i); touched.insert(k);
            }
        }

        for(uint64_t k : touched){
            auto it = freq.find(k);
            if(it != freq.end()) pq.push({it->second, k});
        }

        if(merges.size() % 200 == 0)
            cout << merges.size() << "  " << id2token[newId]
                 << "  freq=" << bestCount
                 << "  vocab=" << id2token.size() << endl;
    }

    /* ---------- 保存 ---------- */
    {
        ofstream f("train/bpe_vocab.txt");
        f << id2token.size() << "\n";
        for(size_t i = 0; i < id2token.size(); i++)
            f << i << ' ' << id2token[i] << "\n";
    }
    {
        ofstream f("train/bpe_merges.txt");
        f << merges.size() << "\n";
        for(auto &m : merges)
            f << m.first.first << ' ' << m.first.second << ' ' << m.second << "\n";
    }

    long long after = 0;
    for(int v : tok) if(v >= 0) after++;
    cout << "\nfinal vocab: " << id2token.size() << endl;
    cout << "merges: " << merges.size() << endl;
    cout << "tokens: " << total << " -> " << after << endl;
    cout << "compression: " << (double)total / after << endl;
    return 0;
}