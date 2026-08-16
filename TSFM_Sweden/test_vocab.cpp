//  bpe_test.cpp
//  读取 bpe_vocab.txt 和 bpe_merges.txt,测试 encode/decode 正确性

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <set>
#include <random>
using namespace std;

/*      ↓↓↓ 以下必须和 bpe.cpp 完全一致 ↓↓↓      */

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

/*      ↑↑↑ 以上必须和 bpe.cpp 完全一致 ↑↑↑      */

vector<string> id2token;
unordered_map<string,int> token2id;
map<pair<int,int>,int> pairRank;
vector<pair<int,int>> rankPair;
vector<int> rankNew;
int UNK = 0;

bool loadVocab(const string &path){
    ifstream f(path);
    if(!f){ cout << "无法打开 " << path << endl; return false; }
    int size;
    f >> size;
    f.ignore();
    for(int i = 0;i < size;i++){
        string line;
        if(!getline(f, line)){ cout << "vocab 文件行数不足,读到第 " << i << " 行" << endl; return false; }
        size_t sp = line.find(' ');
        if(sp == string::npos){ cout << "vocab 第 " << i << " 行格式错误" << endl; return false; }
        int id = stoi(line.substr(0, sp));
        string tok = line.substr(sp + 1);
        if(id != i){ cout << "vocab id 不连续: 期望 " << i << " 实际 " << id << endl; return false; }
        id2token.push_back(tok);
        token2id[tok] = i;
    }
    f.close();
    return true;
}

bool loadMerges(const string &path){
    ifstream f(path);
    if(!f){ cout << "无法打开 " << path << endl; return false; }
    int m;
    f >> m;
    for(int i = 0;i < m;i++){
        int a, b, nid;
        if(!(f >> a >> b >> nid)){ cout << "merges 文件行数不足,读到第 " << i << " 行" << endl; return false; }
        pairRank[{a,b}] = i;
        rankPair.push_back({a,b});
        rankNew.push_back(nid);
    }
    f.close();
    return true;
}

vector<int> encodeSegment(vector<int> seg){
    while(true){
        int bestRank = -1;
        for(int i = 0;i + 1 < (int)seg.size();i++){
            map<pair<int,int>,int>::iterator it = pairRank.find({seg[i], seg[i+1]});
            if(it == pairRank.end()) continue;
            if(bestRank < 0 || it->second < bestRank) bestRank = it->second;
        }
        if(bestRank < 0) break;

        int a = rankPair[bestRank].first;
        int b = rankPair[bestRank].second;
        int newId = rankNew[bestRank];

        vector<int> out;
        for(int i = 0;i < (int)seg.size();){
            if(i + 1 < (int)seg.size() && seg[i]==a && seg[i+1]==b){
                out.push_back(newId); i += 2;
            }else{
                out.push_back(seg[i]); i += 1;
            }
        }
        seg = out;
    }
    return seg;
}

vector<int> encode(const string &text){
    vector<vector<string>> segsStr;
    split(text, segsStr);

    vector<int> res;
    for(int i = 0;i < (int)segsStr.size();i++){
        vector<int> seg;
        for(int j = 0;j < (int)segsStr[i].size();j++){
            unordered_map<string,int>::const_iterator it = token2id.find(segsStr[i][j]);
            seg.push_back(it != token2id.end() ? it->second : UNK);
        }
        seg = encodeSegment(seg);
        for(int j = 0;j < (int)seg.size();j++) res.push_back(seg[j]);
    }
    return res;
}

string decode(const vector<int> &ids){
    string res;
    for(int i = 0;i < (int)ids.size();i++) res += id2token[ids[i]];
    return res;
}

int uft8_count(const string &x){
    int cnt = 0;
    for(size_t i = 0;i < x.size();){
        unsigned char t = x[i];
        i += (t < 128) ? 1 : (t < 224) ? 2 : (t < 240) ? 3 : 4;
        cnt++;
    }
    return cnt;
}

// 把切好的段拼回纯文本(去掉了空格换行,这是 split 的行为)
string segsToString(const vector<vector<string>> &segs, int from, int to){
    string res;
    for(int s = from;s < to && s < (int)segs.size();s++)
        for(int j = 0;j < (int)segs[s].size();j++)
            res += segs[s][j];
    return res;
}

int main(){
    cout << "===== 加载 =====" << endl;
    if(!loadVocab("train/bpe_vocab.txt")) return 1;
    if(!loadMerges("train/bpe_merges.txt")) return 1;

    unordered_map<string,int>::iterator itUnk = token2id.find("<|unk|>");
    if(itUnk == token2id.end()){ cout << "词表里找不到 <|unk|>" << endl; return 1; }
    UNK = itUnk->second;

    cout << "vocab: " << id2token.size() << endl;
    cout << "merges: " << rankPair.size() << endl;
    cout << "UNK id: " << UNK << endl;

    cout << "\n===== TEST 1: 切分效果 =====" << endl;
    {
        vector<string> samples = {
            "程心站在飞船的舷窗前，看着远处的太阳。",
            "罗辑对三体世界说，我们已经知道了黑暗森林法则。",
            "云天明送给程心一颗星星。",
            "这是一个从未出现过的陌生句子，用来测试泛化。"
        };
        for(int k = 0;k < (int)samples.size();k++){
            vector<int> ids = encode(samples[k]);
            cout << "字 " << uft8_count(samples[k]) << " -> token " << ids.size() << " : ";
            for(int i = 0;i < (int)ids.size();i++) cout << "[" << id2token[ids[i]] << "]";
            cout << endl;
        }
    }

    cout << "\n===== TEST 2: 重叠处理 =====" << endl;
    {
        // 构造一个人工场景:找一条 (a,a) 形式的 merge 规则,如果没有就跳过
        bool found = false;
        for(int i = 0;i < (int)rankPair.size();i++){
            if(rankPair[i].first == rankPair[i].second){
                int a = rankPair[i].first;
                vector<int> seg = {a,a,a,a,a};
                vector<int> out = encodeSegment(seg);
                cout << "token '" << id2token[a] << "' x5 -> " << out.size() << " 个: ";
                for(int k = 0;k < (int)out.size();k++) cout << "[" << id2token[out[k]] << "]";
                cout << endl;
                found = true;
                break;
            }
        }
        if(!found) cout << "词表里没有 (a,a) 型合并规则,跳过" << endl;
    }

    cout << "\n===== TEST 3: 往返一致性 =====" << endl;
    {
        ifstream tf("train/train.txt");
        if(!tf){ cout << "无法打开 train/train.txt" << endl; return 1; }
        stringstream buf;
        buf << tf.rdbuf();
        string text = buf.str();
        tf.close();

        vector<vector<string>> segs;
        split(text, segs);
        cout << "语料段数: " << segs.size() << endl;

        mt19937 gen(12345);
        uniform_int_distribution<int> ds(0, (int)segs.size() - 21);
        int fail = 0;
        const int TRIALS = 2000;
        for(int trial = 0;trial < TRIALS;trial++){
            int start = ds(gen);
            string orig = segsToString(segs, start, start + 15);
            if(orig.empty()) continue;

            vector<int> ids = encode(orig);
            string back = decode(ids);
            if(back != orig){
                if(fail < 3){
                    cout << "  FAIL 原文: " << orig << endl;
                    cout << "       还原: " << back << endl;
                }
                fail++;
            }
        }
        cout << "测试 " << TRIALS << " 个片段, 失败 " << fail
             << (fail == 0 ? "  ✓" : "  ✗") << endl;

        cout << "\n===== TEST 4: 全文压缩率 =====" << endl;
        long long chars = 0;
        for(int s = 0;s < (int)segs.size();s++) chars += segs[s].size();
        vector<int> allIds = encode(text);
        cout << "字符数: " << chars << endl;
        cout << "token数: " << allIds.size() << endl;
        cout << "压缩率: " << (double)chars / allIds.size() << endl;

        cout << "\n===== TEST 5: 多字 token 占比 =====" << endl;
        int multi = 0;
        for(int i = 0;i < (int)allIds.size();i++)
            if(uft8_count(id2token[allIds[i]]) > 1) multi++;
        cout << "多字 token 占比: " << 100.0 * multi / allIds.size() << "%" << endl;

        cout << "\n===== 最长的 20 个 token =====" << endl;
        vector<pair<int,int>> lens;
        for(int i = 0;i < (int)id2token.size();i++)
            lens.push_back({uft8_count(id2token[i]), i});
        sort(lens.rbegin(), lens.rend());
        for(int i = 0;i < 20 && i < (int)lens.size();i++)
            cout << id2token[lens[i].second] << " (" << lens[i].first << "字)  ";
        cout << endl;
    }

    return 0;
}