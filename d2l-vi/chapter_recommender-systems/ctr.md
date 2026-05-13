# Hệ thống gợi ý giàu đặc trưng

Dữ liệu tương tác là chỉ báo cơ bản nhất về sở thích và mối quan tâm của người dùng. Nó đóng vai trò quan trọng trong các mô hình đã giới thiệu trước đó. Tuy nhiên, dữ liệu tương tác thường cực kỳ thưa và đôi khi có thể nhiễu. Để xử lý vấn đề này, chúng ta có thể tích hợp thông tin phụ trợ như đặc trưng của vật phẩm, hồ sơ người dùng, và thậm chí ngữ cảnh mà tương tác xảy ra vào mô hình gợi ý. Việc tận dụng các đặc trưng này hữu ích cho việc đưa ra gợi ý, vì chúng có thể là bộ dự đoán hiệu quả cho sở thích của người dùng, đặc biệt khi dữ liệu tương tác còn thiếu. Do đó, điều thiết yếu là các mô hình gợi ý cũng phải có khả năng xử lý những đặc trưng đó và đem lại cho mô hình một mức độ nhận biết nội dung/ngữ cảnh. Để minh họa loại mô hình gợi ý này, chúng ta giới thiệu một tác vụ khác về tỷ lệ nhấp (click-through rate, CTR) cho gợi ý quảng cáo trực tuyến [McMahan.Holt.Sculley.ea.2013] và trình bày một bộ dữ liệu quảng cáo ẩn danh. Các dịch vụ quảng cáo nhắm mục tiêu đã thu hút sự chú ý rộng rãi và thường được diễn đạt như các engine gợi ý. Việc gợi ý quảng cáo phù hợp với thị hiếu và mối quan tâm cá nhân của người dùng là quan trọng để cải thiện tỷ lệ nhấp.


Các nhà tiếp thị số sử dụng quảng cáo trực tuyến để hiển thị quảng cáo cho khách hàng. Tỷ lệ nhấp là một metric đo số lượt nhấp mà nhà quảng cáo nhận được trên quảng cáo của họ so với số lượt hiển thị, và được biểu diễn dưới dạng phần trăm tính theo công thức:

$$ \textrm{CTR} = \frac{\#\textrm{Clicks}} {\#\textrm{Impressions}} \times 100 \% .$$

Tỷ lệ nhấp là một tín hiệu quan trọng cho biết hiệu quả của các thuật toán dự đoán. Dự đoán tỷ lệ nhấp là tác vụ dự đoán khả năng một thứ gì đó trên một trang web sẽ được nhấp. Các mô hình dự đoán CTR không chỉ có thể được sử dụng trong hệ thống quảng cáo nhắm mục tiêu, mà còn trong các hệ thống gợi ý vật phẩm nói chung (ví dụ, phim, tin tức, sản phẩm), chiến dịch email, và thậm chí cả công cụ tìm kiếm. Nó cũng liên quan chặt chẽ đến mức độ hài lòng của người dùng, tỷ lệ chuyển đổi, và có thể hữu ích trong việc đặt mục tiêu chiến dịch vì nó giúp nhà quảng cáo đặt kỳ vọng thực tế.

```python
#@tab mxnet
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```

## Một bộ dữ liệu quảng cáo trực tuyến

Với những tiến bộ đáng kể của Internet và công nghệ di động, quảng cáo trực tuyến đã trở thành một nguồn thu nhập quan trọng và tạo ra phần lớn doanh thu trong ngành Internet. Việc hiển thị các quảng cáo liên quan hoặc các quảng cáo khơi gợi sự quan tâm của người dùng là quan trọng, để những khách truy cập thông thường có thể được chuyển đổi thành khách hàng trả tiền. Bộ dữ liệu chúng ta giới thiệu là một bộ dữ liệu quảng cáo trực tuyến. Nó gồm 34 trường, trong đó cột đầu tiên biểu diễn biến mục tiêu cho biết quảng cáo có được nhấp (1) hay không (0). Tất cả các cột còn lại là đặc trưng phân loại. Các cột có thể biểu diễn id quảng cáo, id trang web hoặc ứng dụng, id thiết bị, thời gian, hồ sơ người dùng, v.v. Ngữ nghĩa thực sự của các đặc trưng không được công bố do quá trình ẩn danh hóa và mối quan tâm về quyền riêng tư.

Đoạn mã sau tải bộ dữ liệu từ máy chủ của chúng ta và lưu nó vào thư mục dữ liệu cục bộ.

```python
#@tab mxnet
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```

Có một tập huấn luyện và một tập kiểm tra, lần lượt gồm 15000 và 3000 mẫu/dòng.

## Wrapper bộ dữ liệu

Để thuận tiện cho việc nạp dữ liệu, chúng ta triển khai một `CTRDataset`, nạp bộ dữ liệu quảng cáo từ tệp CSV và có thể được dùng bởi `DataLoader`.

```python
#@tab mxnet
class CTRDataset(gluon.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                label = np.float32([0, 0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat_v: idx for idx, feat_v in enumerate(feat_values)}
                                for i, feat_values in feat_mapper.items()}
            self.defaults = {i: len(feat_values) for i, feat_values in feat_mapper.items()}
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array((0, *np.cumsum(self.field_dims).asnumpy()
                                 [:-1]))
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                         for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, self.data[idx]['y']
```

Ví dụ sau nạp dữ liệu huấn luyện và in bản ghi đầu tiên.

```python
#@tab mxnet
train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
train_data[0]
```

Như có thể thấy, tất cả 34 trường đều là đặc trưng phân loại. Mỗi giá trị biểu diễn chỉ mục one-hot của mục tương ứng. Nhãn $0$ có nghĩa là nó không được nhấp. `CTRDataset` này cũng có thể được dùng để nạp các bộ dữ liệu khác như [bộ dữ liệu](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) Criteo display advertising challenge và [bộ dữ liệu](https://www.kaggle.com/c/avazu-ctr-prediction) Avazu click-through rate prediction.

## Tóm tắt
* Tỷ lệ nhấp là một metric quan trọng được dùng để đo hiệu quả của các hệ thống quảng cáo và hệ thống gợi ý.
* Dự đoán tỷ lệ nhấp thường được chuyển thành một bài toán phân loại nhị phân. Mục tiêu là dự đoán liệu một quảng cáo/vật phẩm có được nhấp hay không dựa trên các đặc trưng đã cho.

## Bài tập

* Bạn có thể nạp bộ dữ liệu Criteo và Avazu bằng `CTRDataset` đã cung cấp không? Cần lưu ý rằng bộ dữ liệu Criteo gồm các đặc trưng giá trị thực, vì vậy bạn có thể phải sửa mã một chút.
