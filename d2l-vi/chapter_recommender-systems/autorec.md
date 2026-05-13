# AutoRec: Dự đoán rating bằng autoencoder

Mặc dù mô hình phân rã ma trận đạt hiệu năng khá tốt trên tác vụ dự đoán rating, về bản chất nó là một mô hình tuyến tính. Vì vậy, các mô hình như vậy không có khả năng nắm bắt những quan hệ phi tuyến phức tạp và tinh vi có thể dự đoán sở thích của người dùng. Trong phần này, chúng ta giới thiệu AutoRec, một mô hình lọc cộng tác bằng mạng nơ-ron phi tuyến [Sedhain.Menon.Sanner.ea.2015]. Nó đồng nhất lọc cộng tác (collaborative filtering, CF) với một kiến trúc autoencoder và hướng tới việc tích hợp các biến đổi phi tuyến vào CF trên cơ sở phản hồi tường minh. Mạng nơ-ron đã được chứng minh là có khả năng xấp xỉ bất kỳ hàm liên tục nào, nhờ đó phù hợp để xử lý hạn chế của phân rã ma trận và làm phong phú khả năng biểu đạt của phân rã ma trận.

Một mặt, AutoRec có cùng cấu trúc với một autoencoder, gồm một tầng đầu vào, một tầng ẩn và một tầng tái tạo (đầu ra). Autoencoder là một mạng nơ-ron học cách sao chép đầu vào của nó sang đầu ra nhằm mã hóa đầu vào thành các biểu diễn ẩn (và thường là số chiều thấp). Trong AutoRec, thay vì embedding tường minh người dùng/vật phẩm vào không gian số chiều thấp, nó sử dụng cột/hàng của ma trận tương tác làm đầu vào, rồi tái tạo ma trận tương tác ở tầng đầu ra.

Mặt khác, AutoRec khác với một autoencoder truyền thống: thay vì học các biểu diễn ẩn, AutoRec tập trung vào việc học/tái tạo tầng đầu ra. Nó dùng một ma trận tương tác được quan sát một phần làm đầu vào, với mục tiêu tái tạo một ma trận rating đầy đủ. Đồng thời, các phần tử thiếu của đầu vào được điền ở tầng đầu ra thông qua quá trình tái tạo cho mục đích gợi ý.

Có hai biến thể của AutoRec: dựa trên người dùng và dựa trên vật phẩm. Để ngắn gọn, ở đây chúng ta chỉ giới thiệu AutoRec dựa trên vật phẩm. AutoRec dựa trên người dùng có thể được suy ra tương tự.


## Mô hình

Gọi $\mathbf{R}_{*i}$ là cột thứ $i^\textrm{th}$ của ma trận rating, trong đó các rating chưa biết mặc định được đặt bằng 0. Kiến trúc nơ-ron được định nghĩa là:

$$
h(\mathbf{R}_{*i}) = f(\mathbf{W} \cdot g(\mathbf{V} \mathbf{R}_{*i} + \mu) + b)
$$

trong đó $f(\cdot)$ và $g(\cdot)$ biểu diễn các hàm kích hoạt, $\mathbf{W}$ và $\mathbf{V}$ là các ma trận trọng số, còn $\mu$ và $b$ là các độ lệch. Gọi $h( \cdot )$ là toàn bộ mạng của AutoRec. Đầu ra $h(\mathbf{R}_{*i})$ là phần tái tạo của cột thứ $i^\textrm{th}$ trong ma trận rating.

Hàm mục tiêu sau nhằm cực tiểu hóa lỗi tái tạo:

$$
\underset{\mathbf{W},\mathbf{V},\mu, b}{\mathrm{argmin}} \sum_{i=1}^M{\parallel \mathbf{R}_{*i} - h(\mathbf{R}_{*i})\parallel_{\mathcal{O}}^2} +\lambda(\| \mathbf{W} \|_F^2 + \| \mathbf{V}\|_F^2)
$$

trong đó $\| \cdot \|_{\mathcal{O}}$ có nghĩa là chỉ xét đóng góp của các rating đã quan sát được, tức là chỉ các trọng số gắn với đầu vào đã quan sát mới được cập nhật trong quá trình lan truyền ngược.

```python
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx

npx.set_np()
```

## Triển khai mô hình

Một autoencoder điển hình gồm một encoder và một decoder. Encoder chiếu đầu vào sang các biểu diễn ẩn, còn decoder ánh xạ tầng ẩn sang tầng tái tạo. Chúng ta làm theo thực hành này và tạo encoder cùng decoder bằng các tầng fully connected. Kích hoạt của encoder mặc định được đặt là `sigmoid`, còn decoder không áp dụng kích hoạt. Dropout được thêm vào sau biến đổi mã hóa để giảm overfitting. Gradient của các đầu vào chưa quan sát được che lại để đảm bảo rằng chỉ các rating đã quan sát mới đóng góp vào quá trình học của mô hình.

```python
#@tab mxnet
class AutoRec(nn.Block):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Dense(num_hidden, activation='sigmoid',
                                use_bias=True)
        self.decoder = nn.Dense(num_users, use_bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(self.encoder(input))
        pred = self.decoder(hidden)
        if autograd.is_training():  # Mask the gradient during training
            return pred * np.sign(input)
        else:
            return pred
```

## Triển khai lại bộ đánh giá

Vì đầu vào và đầu ra đã thay đổi, chúng ta cần triển khai lại hàm đánh giá, trong khi vẫn dùng RMSE làm thước đo độ chính xác.

```python
#@tab mxnet
def evaluator(network, inter_matrix, test_data, devices):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, devices, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # Calculate the test RMSE
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                   / np.sum(np.sign(test_data)))
    return float(rmse)
```

## Huấn luyện và đánh giá mô hình

Bây giờ, hãy huấn luyện và đánh giá AutoRec trên bộ dữ liệu MovieLens. Chúng ta có thể thấy rõ rằng RMSE kiểm tra thấp hơn mô hình phân rã ma trận, xác nhận hiệu quả của mạng nơ-ron trong tác vụ dự đoán rating.

```python
#@tab mxnet
devices = d2l.try_all_gpus()
# Load the MovieLens 100K dataset
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items)
_, _, _, train_inter_mat = d2l.load_data_ml100k(train_data, num_users,
                                                num_items)
_, _, _, test_inter_mat = d2l.load_data_ml100k(test_data, num_users,
                                               num_items)
train_iter = gluon.data.DataLoader(train_inter_mat, shuffle=True,
                                   last_batch="rollover", batch_size=256,
                                   num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(np.array(train_inter_mat), shuffle=False,
                                  last_batch="keep", batch_size=1024,
                                  num_workers=d2l.get_dataloader_workers())
# Model initialization, training, and evaluation
net = AutoRec(500, num_users)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 25, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
d2l.train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices, evaluator, inter_mat=test_inter_mat)
```

## Tóm tắt

* Chúng ta có thể diễn đạt thuật toán phân rã ma trận bằng autoencoder, đồng thời tích hợp các tầng phi tuyến và điều chuẩn dropout.
* Các thí nghiệm trên bộ dữ liệu MovieLens 100K cho thấy AutoRec đạt hiệu năng tốt hơn phân rã ma trận.


## Bài tập

* Thay đổi số chiều ẩn của AutoRec để xem tác động của nó lên hiệu năng mô hình.
* Thử thêm nhiều tầng ẩn hơn. Điều đó có giúp cải thiện hiệu năng mô hình không?
* Bạn có thể tìm được một tổ hợp hàm kích hoạt decoder và encoder tốt hơn không?
