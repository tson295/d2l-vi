# Deep Factorization Machines

Học các tổ hợp đặc trưng hiệu quả là yếu tố then chốt cho thành công của tác vụ dự đoán tỷ lệ nhấp. Factorization machines mô hình hóa tương tác đặc trưng theo một khuôn mẫu tuyến tính (ví dụ, tương tác song tuyến tính). Điều này thường không đủ cho dữ liệu thực tế, nơi các cấu trúc giao cắt đặc trưng nội tại thường rất phức tạp và phi tuyến. Tệ hơn nữa, trong thực hành, các tương tác đặc trưng bậc hai thường được dùng trong factorization machines. Về mặt lý thuyết, có thể mô hình hóa các bậc cao hơn của tổ hợp đặc trưng bằng factorization machines, nhưng thường không được áp dụng do bất ổn định số và độ phức tạp tính toán cao.

Một giải pháp hiệu quả là sử dụng mạng nơ-ron sâu. Mạng nơ-ron sâu rất mạnh trong học biểu diễn đặc trưng và có tiềm năng học các tương tác đặc trưng tinh vi. Vì vậy, việc tích hợp mạng nơ-ron sâu vào factorization machines là tự nhiên. Thêm các tầng biến đổi phi tuyến vào factorization machines đem lại cho nó khả năng mô hình hóa cả các tổ hợp đặc trưng bậc thấp lẫn bậc cao. Hơn nữa, các cấu trúc nội tại phi tuyến từ đầu vào cũng có thể được nắm bắt bằng mạng nơ-ron sâu. Trong phần này, chúng ta sẽ giới thiệu một mô hình đại diện có tên deep factorization machines (DeepFM) [Guo.Tang.Ye.ea.2017], kết hợp FM và mạng nơ-ron sâu.


## Kiến trúc mô hình

DeepFM gồm một thành phần FM và một thành phần sâu, được tích hợp trong một cấu trúc song song. Thành phần FM giống với factorization machines bậc 2, được dùng để mô hình hóa các tương tác đặc trưng bậc thấp. Thành phần sâu là một MLP được dùng để nắm bắt các tương tác đặc trưng bậc cao và tính phi tuyến. Hai thành phần này chia sẻ cùng đầu vào/embedding và các đầu ra của chúng được cộng lại làm dự đoán cuối cùng. Cần chỉ ra rằng tinh thần của DeepFM tương tự kiến trúc Wide \& Deep, vốn có thể nắm bắt cả khả năng ghi nhớ và tổng quát hóa. Lợi thế của DeepFM so với mô hình Wide \& Deep là nó giảm công sức kỹ thuật đặc trưng thủ công bằng cách tự động nhận diện các tổ hợp đặc trưng.

Để ngắn gọn, chúng ta bỏ qua mô tả của thành phần FM và ký hiệu đầu ra là $\hat{y}^{(FM)}$. Độc giả có thể tham khảo phần trước để biết thêm chi tiết. Gọi $\mathbf{e}_i \in \mathbb{R}^{k}$ là vector đặc trưng tiềm ẩn của trường thứ $i^\textrm{th}$. Đầu vào của thành phần sâu là phép nối các embedding đặc của tất cả trường, được tra cứu bằng đầu vào đặc trưng phân loại thưa, ký hiệu là:

$$
\mathbf{z}^{(0)}  = [\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_f],
$$

trong đó $f$ là số trường. Sau đó, nó được đưa vào mạng nơ-ron sau:

$$
\mathbf{z}^{(l)}  = \alpha(\mathbf{W}^{(l)}\mathbf{z}^{(l-1)} + \mathbf{b}^{(l)}),
$$

trong đó $\alpha$ là hàm kích hoạt. $\mathbf{W}_{l}$ và $\mathbf{b}_{l}$ là trọng số và độ lệch tại tầng thứ $l^\textrm{th}$. Gọi $y_{DNN}$ là đầu ra của phần dự đoán. Dự đoán cuối cùng của DeepFM là tổng các đầu ra từ cả FM và DNN. Do đó ta có:

$$
\hat{y} = \sigma(\hat{y}^{(FM)} + \hat{y}^{(DNN)}),
$$

trong đó $\sigma$ là hàm sigmoid. Kiến trúc của DeepFM được minh họa dưới đây.
![Minh họa mô hình DeepFM](../img/rec-deepfm.svg)

Cần lưu ý rằng DeepFM không phải là cách duy nhất để kết hợp mạng nơ-ron sâu với FM. Chúng ta cũng có thể thêm các tầng phi tuyến lên trên các tương tác đặc trưng [He.Chua.2017].

```python
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## Triển khai DeepFM
Triển khai DeepFM tương tự FM. Chúng ta giữ nguyên phần FM và dùng một khối MLP với `relu` làm hàm kích hoạt. Dropout cũng được dùng để điều chuẩn mô hình. Số nơ-ron của MLP có thể được điều chỉnh bằng siêu tham số `mlp_dims`.

```python
#@tab mxnet
class DeepFM(nn.Block):
    def __init__(self, field_dims, num_factors, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)
        input_dim = self.embed_output_dim = len(field_dims) * num_factors
        self.mlp = nn.Sequential()
        for dim in mlp_dims:
            self.mlp.add(nn.Dense(dim, 'relu', True, in_units=input_dim))
            self.mlp.add(nn.Dropout(rate=drop_rate))
            input_dim = dim
        self.mlp.add(nn.Dense(in_units=input_dim, units=1))

    def forward(self, x):
        embed_x = self.embedding(x)
        square_of_sum = np.sum(embed_x, axis=1) ** 2
        sum_of_square = np.sum(embed_x ** 2, axis=1)
        inputs = np.reshape(embed_x, (-1, self.embed_output_dim))
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True) \
            + self.mlp(inputs)
        x = npx.sigmoid(x)
        return x
```

## Huấn luyện và đánh giá mô hình
Quy trình nạp dữ liệu giống như FM. Chúng ta đặt thành phần MLP của DeepFM là một mạng dense ba tầng với cấu trúc kim tự tháp (30-20-10). Tất cả siêu tham số khác giữ nguyên như FM.

```python
#@tab mxnet
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
field_dims = train_data.field_dims
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
devices = d2l.try_all_gpus()
net = DeepFM(field_dims, num_factors=10, mlp_dims=[30, 20, 10])
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.01, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

So với FM, DeepFM hội tụ nhanh hơn và đạt hiệu năng tốt hơn.

## Tóm tắt

* Tích hợp mạng nơ-ron vào FM cho phép nó mô hình hóa các tương tác phức tạp và bậc cao.
* DeepFM vượt trội hơn FM gốc trên bộ dữ liệu quảng cáo.

## Bài tập

* Thay đổi cấu trúc của MLP để kiểm tra tác động của nó lên hiệu năng mô hình.
* Đổi bộ dữ liệu sang Criteo và so sánh với mô hình FM gốc.
