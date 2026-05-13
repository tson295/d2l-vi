# Factorization Machines

Factorization machines (FM), do Rendle.2010 đề xuất, là một thuật toán có giám sát có thể được dùng cho các tác vụ phân loại, hồi quy và xếp hạng. Nó nhanh chóng được chú ý và trở thành một phương pháp phổ biến, có ảnh hưởng trong việc đưa ra dự đoán và gợi ý. Đặc biệt, nó là một sự tổng quát hóa của mô hình hồi quy tuyến tính và mô hình phân rã ma trận. Hơn nữa, nó gợi nhớ đến máy vector hỗ trợ với kernel đa thức. Các điểm mạnh của factorization machines so với hồi quy tuyến tính và phân rã ma trận là: (1) nó có thể mô hình hóa các tương tác biến $\chi$-chiều, trong đó $\chi$ là bậc đa thức và thường được đặt bằng hai. (2) Một thuật toán tối ưu hóa nhanh gắn với factorization machines có thể giảm thời gian tính toán đa thức xuống độ phức tạp tuyến tính, khiến nó cực kỳ hiệu quả, đặc biệt với đầu vào thưa số chiều cao. Vì những lý do này, factorization machines được sử dụng rộng rãi trong quảng cáo hiện đại và gợi ý sản phẩm. Các chi tiết kỹ thuật và triển khai được mô tả dưới đây.


## Factorization Machines bậc 2

Một cách hình thức, gọi $x \in \mathbb{R}^d$ là vector đặc trưng của một mẫu, và $y$ là nhãn tương ứng, có thể là nhãn giá trị thực hoặc nhãn lớp như lớp nhị phân "click/non-click". Mô hình cho một factorization machine bậc hai được định nghĩa là:

$$
\hat{y}(x) = \mathbf{w}_0 + \sum_{i=1}^d \mathbf{w}_i x_i + \sum_{i=1}^d\sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j
$$

trong đó $\mathbf{w}_0 \in \mathbb{R}$ là độ lệch toàn cục; $\mathbf{w} \in \mathbb{R}^d$ biểu thị trọng số của biến thứ i; $\mathbf{V} \in \mathbb{R}^{d\times k}$ biểu diễn các embedding đặc trưng; $\mathbf{v}_i$ biểu diễn hàng thứ $i^\textrm{th}$ của $\mathbf{V}$; $k$ là số chiều của các nhân tố tiềm ẩn; $\langle\cdot, \cdot \rangle$ là tích vô hướng của hai vector. $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ mô hình hóa tương tác giữa đặc trưng thứ $i^\textrm{th}$ và thứ $j^\textrm{th}$. Một số tương tác đặc trưng có thể được hiểu dễ dàng nên có thể được thiết kế bởi chuyên gia. Tuy nhiên, phần lớn các tương tác đặc trưng khác bị ẩn trong dữ liệu và khó nhận diện. Vì vậy, mô hình hóa tự động các tương tác đặc trưng có thể giảm đáng kể công sức trong kỹ thuật đặc trưng. Rõ ràng hai hạng đầu tiên tương ứng với mô hình hồi quy tuyến tính, còn hạng cuối là một mở rộng của mô hình phân rã ma trận. Nếu đặc trưng $i$ biểu diễn một vật phẩm và đặc trưng $j$ biểu diễn một người dùng, hạng thứ ba chính là tích vô hướng giữa embedding người dùng và vật phẩm. Cần lưu ý rằng FM cũng có thể tổng quát hóa lên các bậc cao hơn (bậc > 2). Tuy nhiên, độ ổn định số có thể làm suy yếu khả năng tổng quát hóa.


## Một tiêu chí tối ưu hóa hiệu quả

Tối ưu factorization machines theo cách trực tiếp dẫn đến độ phức tạp $\mathcal{O}(kd^2)$ vì tất cả tương tác từng cặp đều cần được tính. Để giải quyết sự kém hiệu quả này, chúng ta có thể tổ chức lại hạng thứ ba của FM, nhờ đó giảm đáng kể chi phí tính toán và dẫn đến độ phức tạp thời gian tuyến tính ($\mathcal{O}(kd)$). Phép biến đổi lại hạng tương tác từng cặp như sau:

$$
\begin{aligned}
&\sum_{i=1}^d \sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j \\
 &= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d\langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j - \frac{1}{2}\sum_{i=1}^d \langle\mathbf{v}_i, \mathbf{v}_i\rangle x_i x_i \\
 &= \frac{1}{2} \big (\sum_{i=1}^d \sum_{j=1}^d \sum_{l=1}^k\mathbf{v}_{i, l} \mathbf{v}_{j, l} x_i x_j - \sum_{i=1}^d \sum_{l=1}^k \mathbf{v}_{i, l} \mathbf{v}_{i, l} x_i x_i \big)\\
 &=  \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i) (\sum_{j=1}^d \mathbf{v}_{j, l}x_j) - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2 \big ) \\
 &= \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i)^2 - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2)
 \end{aligned}
$$

Với phép biến đổi này, độ phức tạp của mô hình giảm mạnh. Hơn nữa, với các đặc trưng thưa, chỉ các phần tử khác không cần được tính, do đó độ phức tạp tổng thể tuyến tính theo số đặc trưng khác không.

Để học mô hình FM, chúng ta có thể dùng mất mát MSE cho tác vụ hồi quy, mất mát cross-entropy cho tác vụ phân loại, và mất mát BPR cho tác vụ xếp hạng. Các optimizer chuẩn như stochastic gradient descent và Adam đều khả thi cho việc tối ưu hóa.

```python
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## Triển khai mô hình
Đoạn mã sau triển khai factorization machines. Có thể thấy rõ rằng FM gồm một khối hồi quy tuyến tính và một khối tương tác đặc trưng hiệu quả. Chúng ta áp dụng hàm sigmoid lên điểm cuối cùng vì xem dự đoán CTR là một tác vụ phân loại.

```python
#@tab mxnet
class FM(nn.Block):
    def __init__(self, field_dims, num_factors):
        super(FM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)

    def forward(self, x):
        square_of_sum = np.sum(self.embedding(x), axis=1) ** 2
        sum_of_square = np.sum(self.embedding(x) ** 2, axis=1)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
        x = npx.sigmoid(x)
        return x
```

## Nạp bộ dữ liệu quảng cáo
Chúng ta dùng wrapper dữ liệu CTR từ phần trước để nạp bộ dữ liệu quảng cáo trực tuyến.

```python
#@tab mxnet
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
```

## Huấn luyện mô hình
Sau đó, chúng ta huấn luyện mô hình. Tốc độ học mặc định được đặt là 0.02 và kích thước embedding được đặt là 20. Optimizer `Adam` và mất mát `SigmoidBinaryCrossEntropyLoss` được dùng cho huấn luyện mô hình.

```python
#@tab mxnet
devices = d2l.try_all_gpus()
net = FM(train_data.field_dims, num_factors=20)
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.02, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## Tóm tắt

* FM là một khung tổng quát có thể được áp dụng cho nhiều tác vụ như hồi quy, phân loại và xếp hạng.
* Tương tác/giao cắt đặc trưng là quan trọng đối với các tác vụ dự đoán, và tương tác bậc 2 có thể được mô hình hóa hiệu quả bằng FM.

## Bài tập

* Bạn có thể thử FM trên các bộ dữ liệu khác như Avazu, MovieLens và Criteo không?
* Thay đổi kích thước embedding để kiểm tra tác động của nó lên hiệu năng; bạn có quan sát thấy một mẫu tương tự như trong phân rã ma trận không?
