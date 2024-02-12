class Trie:
    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for _ in range(256)]
        self.values = set()
        self.front = front
    
    def add(self, key: bytes, idx = 0, val=None):
        if idx == len(key):
            val = val if val is not None else key
            self.values.add(val)
            return self
        ch = key[idx]
        if self.to[ch] is None:
            self.to[ch] = Trie(front=self, ch=ch)
        return self.to[ch].add(key, idx+1, val)

    def find_longest(self, key:bytes, idx=0):
        curr = self
        ch = key[idx]
        while curr.to[ch] is not None:
            curr =  curr.to[ch]
            idx += 1
            if curr.values:
                ret = idx, curr.values
            if idx == len(key):
                break
            ch = key[idx]
        return ret

def parse_line(line):
    lspace = line.index(" ")
    rspace = line.rindex(" ")
    token = eval(line[lspace:rspace])
    return int(line[:lspace]), token.encode("utf-8") if isinstance(token, str) else token, int(line[rspace+1:])

class TrieTokenizer:
    def __init__(self, file_name: str):
        self.idx2token = {}
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                idx, token, c = parse_line(line)
                assert isinstance(token, bytes), "token is not in bytes format"
                assert len(token) == c, f"token: {token} length is not equal to c"
                self.idx2token[idx] = token
        self.token2idx = {}
        for k,v in self.idx2token.items():
            self.token2idx[v] = k
        
        self.root = Trie()
        for t, i in self.token2idx.items():
            self.root.add(t, val=(t, i))
    
    def encodeBytes(self, src: bytes):
        idx = 0
        tokens = []
        while idx < len(src):
            _idx = idx
            idx, values = self.root.find_longest(src, idx)
            assert idx != _idx
            _, token = next(iter(values))
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))
    
    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        try: 
            return self.decodeBytes(tokens).decode("utf-8") 
        except:
            raise '\ufffd'
            
if __name__ == "__main__":
    tokenizer = TrieTokenizer("./rwkv5-world-1b/rwkv_vocab_v20230424.txt")
    print(tokenizer.encode("hello world"))