# Dữ liệu Hồi quy Tổng hợp
<a id="sec_synthetic-regression-data"></a>


Machine learning là tất cả về việc trích xuất thông tin từ dữ liệu.
Vì vậy bạn có thể tự hỏi, ta có thể học được gì từ dữ liệu tổng hợp?
Mặc dù ta có thể không thực sự quan tâm đến các mẫu
mà ta đã tự tạo ra trong mô hình sinh dữ liệu nhân tạo,
các tập dữ liệu như vậy vẫn hữu ích cho mục đích giảng dạy,
giúp ta đánh giá các thuộc tính của thuật toán học
và xác nhận rằng các cài đặt của ta hoạt động đúng như mong đợi.
Ví dụ, nếu ta tạo dữ liệu mà các tham số đúng đã biết *trước*,
thì ta có thể kiểm tra liệu mô hình của ta có thực sự khôi phục chúng được không.


```python
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```


## Tạo Tập dữ liệu

Trong ví dụ này, ta sẽ làm việc trong không gian thấp chiều
để ngắn gọn.
Đoạn code sau tạo ra 1000 mẫu
với các đặc trưng 2 chiều được rút ra
từ phân phối chuẩn tiêu chuẩn.
Ma trận thiết kế kết quả $\mathbf{X}$
thuộc $\mathbb{R}^{1000 \times 2}$.
Ta tạo mỗi nhãn bằng cách áp dụng
một hàm tuyến tính *thực tế*,
làm hỏng chúng thông qua nhiễu cộng tính $\boldsymbol{\epsilon}$,
được rút ra độc lập và đồng nhất cho mỗi mẫu:

(**$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \boldsymbol{\epsilon}.$$**)

Để thuận tiện, ta giả định rằng $\boldsymbol{\epsilon}$ được rút ra
từ phân phối chuẩn với trung bình $\mu= 0$
và độ lệch chuẩn $\sigma = 0.01$.
Lưu ý rằng để thiết kế hướng đối tượng
ta thêm code vào phương thức `__init__` của lớp con của `d2l.DataModule` (giới thiệu trong [oo-design-data](#oo-design-data)).
Đây là thực hành tốt để cho phép thiết lập bất kỳ siêu tham số bổ sung nào.
Ta thực hiện điều này với `save_hyperparameters()`.
`batch_size` sẽ được xác định sau.

```python
class SyntheticRegressionData(d2l.DataModule):  
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, 
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        if tab.selected('pytorch') or tab.selected('mxnet'):                
            self.X = d2l.randn(n, len(w))
            noise = d2l.randn(n, 1) * noise
        if tab.selected('tensorflow'):
            self.X = tf.random.normal((n, w.shape[0]))
            noise = tf.random.normal((n, 1)) * noise
        if tab.selected('jax'):
            key = jax.random.PRNGKey(0)
            key1, key2 = jax.random.split(key)
            self.X = jax.random.normal(key1, (n, w.shape[0]))
            noise = jax.random.normal(key2, (n, 1)) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise
```

Dưới đây, ta đặt các tham số thực là $\mathbf{w} = [2, -3.4]^\top$ và $b = 4.2$.
Sau đó ta có thể kiểm tra các tham số ước lượng của mình so với các giá trị *thực* này.

```python
data = SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
```

[**Mỗi hàng trong `features` là một vector trong $\mathbb{R}^2$ và mỗi hàng trong `labels` là một vô hướng.**] Hãy xem mục đầu tiên.

```python
print('features:', data.X[0],'\nlabel:', data.y[0])
```

## Đọc Tập dữ liệu

Huấn luyện các mô hình machine learning thường đòi hỏi nhiều lần duyệt qua tập dữ liệu,
lấy từng minibatch mẫu một.
Dữ liệu này sau đó được dùng để cập nhật mô hình.
Để minh họa cách hoạt động, ta
[**cài đặt phương thức `get_dataloader`,**]
đăng ký nó trong lớp `SyntheticRegressionData` thông qua `add_to_class` (giới thiệu trong [oo-design-utilities](#oo-design-utilities)).
Nó (**nhận kích thước batch, ma trận đặc trưng,
và một vector nhãn, và tạo ra các minibatch có kích thước `batch_size`.**)
Như vậy, mỗi minibatch bao gồm một tuple gồm đặc trưng và nhãn.
Lưu ý rằng ta cần chú ý đến việc ta đang ở chế độ huấn luyện hay kiểm định:
trong chế độ huấn luyện, ta muốn đọc dữ liệu theo thứ tự ngẫu nhiên,
trong khi với kiểm định, khả năng đọc dữ liệu theo thứ tự được định sẵn
có thể quan trọng cho mục đích gỡ lỗi.

```python
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train+self.num_val))
    for i in range(0, len(indices), self.batch_size):
        if tab.selected('mxnet', 'pytorch', 'jax'):
            batch_indices = d2l.tensor(indices[i: i+self.batch_size])
            yield self.X[batch_indices], self.y[batch_indices]
        if tab.selected('tensorflow'):
            j = tf.constant(indices[i : i+self.batch_size])
            yield tf.gather(self.X, j), tf.gather(self.y, j)
```

Để xây dựng một số trực giác, hãy kiểm tra minibatch dữ liệu đầu tiên.
Mỗi minibatch đặc trưng cung cấp cho ta cả kích thước và chiều số của các đặc trưng đầu vào.
Tương tự, minibatch nhãn của ta sẽ có shape tương ứng được cho bởi `batch_size`.

```python
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

Mặc dù có vẻ vô hại, việc gọi
`iter(data.train_dataloader())`
minh họa sức mạnh của thiết kế hướng đối tượng Python.
Lưu ý rằng ta đã thêm một phương thức vào lớp `SyntheticRegressionData`
*sau khi* tạo đối tượng `data`.
Dù vậy, đối tượng vẫn hưởng lợi từ
việc bổ sung chức năng vào lớp *sau khi thực tế*.

Trong suốt quá trình lặp, ta thu được các minibatch khác nhau
cho đến khi toàn bộ tập dữ liệu được duyệt xong (hãy thử điều này).
Trong khi việc lặp cài đặt ở trên tốt cho mục đích giảng dạy,
nó lại kém hiệu quả theo những cách có thể gây rắc rối với các bài toán thực tế.
Ví dụ, nó yêu cầu ta tải tất cả dữ liệu vào bộ nhớ
và ta phải thực hiện nhiều lần truy cập bộ nhớ ngẫu nhiên.
Các bộ lặp tích hợp được cài đặt trong framework deep learning
hiệu quả hơn đáng kể và chúng có thể xử lý
các nguồn như dữ liệu lưu trên file,
dữ liệu nhận qua luồng,
và dữ liệu được tạo hoặc xử lý ngay lúc đó.
Tiếp theo hãy thử cài đặt phương thức tương tự sử dụng các bộ lặp tích hợp.

## Cài đặt Gọn của Bộ Tải Dữ liệu

Thay vì tự viết bộ lặp,
ta có thể [**gọi API có sẵn trong framework để tải dữ liệu.**]
Như trước, ta cần một tập dữ liệu với đặc trưng `X` và nhãn `y`.
Ngoài ra, ta đặt `batch_size` trong bộ tải dữ liệu tích hợp
và để nó xử lý việc xáo trộn các mẫu một cách hiệu quả.


```python
@d2l.add_to_class(d2l.DataModule)  
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    if tab.selected('mxnet'):
        dataset = gluon.data.ArrayDataset(*tensors)
        return gluon.data.DataLoader(dataset, self.batch_size,
                                     shuffle=train)
    if tab.selected('pytorch'):
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)
    if tab.selected('jax'):
        # Use Tensorflow Datasets & Dataloader. JAX or Flax do not provide
        # any dataloading functionality
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(tensors).shuffle(
                buffer_size=shuffle_buffer).batch(self.batch_size))

    if tab.selected('tensorflow'):
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tf.data.Dataset.from_tensor_slices(tensors).shuffle(
            buffer_size=shuffle_buffer).batch(self.batch_size)
```

```python
@d2l.add_to_class(SyntheticRegressionData)  
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)
```

Bộ tải dữ liệu mới hoạt động giống như bộ trước, ngoại trừ nó hiệu quả hơn và có một số chức năng bổ sung.

```python
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

Chẳng hạn, bộ tải dữ liệu do API framework cung cấp
hỗ trợ phương thức `__len__` tích hợp,
vì vậy ta có thể truy vấn độ dài của nó,
tức là số lượng batch.

```python
len(data.train_dataloader())
```

## Tóm tắt

Bộ tải dữ liệu là cách thuận tiện để trừu tượng hóa
quá trình tải và thao tác dữ liệu.
Nhờ đó cùng một *thuật toán* machine learning
có thể xử lý nhiều loại và nguồn dữ liệu khác nhau
mà không cần sửa đổi.
Một trong những điều hay về bộ tải dữ liệu
là chúng có thể được kết hợp.
Chẳng hạn, ta có thể đang tải ảnh
và sau đó có bộ lọc hậu xử lý
cắt chúng hoặc sửa đổi chúng theo những cách khác.
Do đó, bộ tải dữ liệu có thể được dùng
để mô tả toàn bộ pipeline xử lý dữ liệu.

Về bản thân mô hình, mô hình tuyến tính hai chiều
là đơn giản nhất ta có thể gặp.
Nó cho phép ta kiểm tra độ chính xác của các mô hình hồi quy
mà không lo lắng về việc có không đủ lượng dữ liệu
hay một hệ phương trình không xác định.
Ta sẽ tận dụng điều này trong phần tiếp theo.


## Bài tập

1. Điều gì sẽ xảy ra nếu số lượng mẫu không chia hết cho kích thước batch. Bạn sẽ thay đổi hành vi này bằng cách chỉ định một đối số khác khi dùng API của framework như thế nào?
1. Giả sử ta muốn tạo một tập dữ liệu khổng lồ, trong đó cả kích thước vector tham số `w` và số lượng mẫu `num_examples` đều lớn.
    1. Điều gì xảy ra nếu ta không thể chứa tất cả dữ liệu trong bộ nhớ?
    1. Bạn sẽ xáo trộn dữ liệu như thế nào nếu nó được lưu trên đĩa? Nhiệm vụ của bạn là thiết kế một thuật toán *hiệu quả* không đòi hỏi quá nhiều lần đọc hoặc ghi ngẫu nhiên. Gợi ý: [bộ tạo hoán vị giả ngẫu nhiên](https://en.wikipedia.org/wiki/Pseudorandom_permutation) cho phép bạn thiết kế việc xáo trộn lại mà không cần lưu trữ bảng hoán vị một cách rõ ràng [Naor.Reingold.1999].
1. Cài đặt bộ tạo dữ liệu tạo ra dữ liệu mới ngay lúc đó, mỗi khi bộ lặp được gọi.
1. Bạn sẽ thiết kế bộ tạo dữ liệu ngẫu nhiên tạo ra *cùng một* dữ liệu mỗi khi nó được gọi như thế nào?


[Thảo luận](https://discuss.d2l.ai/t/6663)
