# 주의: 기존 코드를 수정하지 마세요.
# 주석을 통해 코드를 설명하는 것을 권장합니다.
# 세미 콜론(;)을 통한 코드 이어 붙이기는 금지입니다.

# Problem 1
# 내부 코드는 오로지 한 줄이어야 합니다.
# Hint: max 함수에 대해 알아보세요.
def p1(X):
    return [max(i, key=lambda x:sum(int(d) for d in str(x))) for i in X]

#Problem 2
# 내부 코드는 최대 세 줄이어야 합니다.
def p2(idx):
    max_idx = max(idx)
    return [[1 if i==x else 0 for i in range(max_idx+1)] for x in filter(lambda x: x>=0,idx)]

# Problem 3
# 내부 코드는 최대 네 줄이어야 합니다.
# Hint: sample과 data의 크기가 클 수 있기 때문에 효율적인 자료구조를 사용하세요.
def p3(sample, data):
    last_name = set(map(lambda x:x[0], data))
    first_name = set(map(lambda x:x[1:], data))
    return [x[0] in last_name and x[1:] in first_name for x in sample]

# Problem 4
# 내부 코드는 최대 여섯 줄이어야 합니다.
def p4(sample, data):
    words = sorted(set(' '.join(map(lambda x:x.lower(), data)).split()))
    w2i = dict(map(lambda x:(x[1], x[0]), enumerate(words)))
    return [[w2i[w] if w in w2i else len(w2i) for w in sent.lower().split()] for sent in sample]

class Matrix2d:
    def __init__(self, data):
        assert len(data) > 0
        self.data = [list(row) for row in data]
        self.shape = (len(data), len(data[0]))
        
    def add(self, x):
        assert self.shape == x.shape
        r, c = self.shape
        for i in range(r):
            for j in range(c):
                self.data[i][j]+=x.data[i][j]
    
    def where(self, func):
        r, c = self.shape
        for i in range(r):
            for j in range(c):
                if func(self.data[i][j]):
                    yield i, j
            
    def __eq__(self, img):
        return img.data == self.data and img.shape == self.shape
    
    def __repr__(self):
        return str([list(map(lambda x: round(x, 4), row)) for row in self.data])

# Problem 5
# p5 함수가 정상 동작 하게끔 아래 클래스를 구현하세요.
class GrayScaleImg(Matrix2d):
    def __init__(self, data):
        assert all(all(0<=x and x<=1 for x in row) for row in data)
        super().__init__(data)
    
    def ImgAdd(self, img):
        # 입력으로 들어오는 img 또한 GrayScaleImg 객체이며 해당 데이터를 현재 데이터에 더하는 메소드
        assert isinstance(img, GrayScaleImg)
        self.add(img)
        for i, j in self.where(lambda x: x < 0):
            self.data[i][j] = 0
        for i, j in self.where(lambda x: x > 1):
            self.data[i][j] = 1
    
    def Transpose(self):
        # 현재 데이터를 transpose 시키는 메소드
        r, c = self.shape
        self.data = [list(map(lambda x:x[i],self.data)) for i in range(c)]
        self.shape = (c, r)
    
    def FixBrightness(self, threshold, ratio):
        # threshold 보다 높은 부분을 ratio 만큼 곱해주는 메소드. ratio는 0과 1사이 값이라는 것이 보장됩니다.
        assert 0 <= ratio and ratio <= 1
        idx = self.where(lambda x: x > threshold)
        for i, j in idx:
            self.data[i][j] *= ratio
    
    def Pad(self):
        # Zero-padding을 추가하는 메소드
        r, c = self.shape
        self.data = list(map(lambda x: [0]+x+[0], self.data))
        self.data = [[0]*(c+2)] + self.data + [[0]*(c+2)]
        self.shape = (r+2, c+2)

        
def p5(data, low, img_to_add, threshold, ratio):
    try:
        result = dict()
        img = GrayScaleImg(data)
        result["lower"] = list(img.where(lambda x: x < low))

        to_add = GrayScaleImg(img_to_add)
        img.ImgAdd(to_add)
        result["add"] = Matrix2d(img.data)

        img.Transpose()
        result["transpose"] = Matrix2d(img.data)

        img.FixBrightness(threshold, ratio)
        result["fix"] = Matrix2d(img.data)

        img.Pad()
        result["pad"] = Matrix2d(img.data)
        return result
    except Exception as e:
        print("[Error]", e)
        return None

# Problem 6
def p6(n):
    m = [1, 1, 1, 0]
    res = [1, 0, 0, 1]
    mod = 10**9+7
    while n:
        if n % 2:
            a,b,c,d = m
            e,f,g,h = res
            res = list(map(lambda x:x%mod,[a*e+b*g,a*f+b*h,c*e+d*g,c*f+d*h]))
        a,b,c,d=m
        m = list(map(lambda x:x%mod, [a*a+b*c,a*b+b*d, c*a+d*c, b*c+d*d]))
        n //= 2
    return res[-2]