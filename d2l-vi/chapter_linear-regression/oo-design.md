# Thiết kế Hướng Đối tượng cho Cài đặt
<a id="sec_oo-design"></a>

Trong phần giới thiệu về hồi quy tuyến tính,
ta đã đi qua các thành phần khác nhau
bao gồm
dữ liệu, mô hình, hàm mất mát,
và thuật toán tối ưu hóa.
Thực ra,
hồi quy tuyến tính là
một trong những mô hình machine learning đơn giản nhất.
Tuy nhiên, việc huấn luyện nó
sử dụng nhiều thành phần tương tự mà các mô hình khác trong cuốn sách này yêu cầu.
Do đó,
trước khi đi sâu vào các chi tiết cài đặt
thì đáng để
thiết kế một số API
mà ta sử dụng xuyên suốt.
Coi các thành phần trong deep learning
là các đối tượng,
ta có thể bắt đầu bằng
định nghĩa các lớp cho các đối tượng này
và các tương tác của chúng.
Thiết kế hướng đối tượng này
cho cài đặt sẽ
đơn giản hóa đáng kể phần trình bày và bạn thậm chí có thể muốn sử dụng nó trong các dự án của mình.


Lấy cảm hứng từ các thư viện mã nguồn mở như [PyTorch Lightning](https://www.pytorchlightning.ai/),
ở cấp độ cao
ta muốn có ba lớp:
(i) `Module` chứa mô hình, hàm mất mát và các phương thức tối ưu hóa;
(ii) `DataModule` cung cấp các bộ tải dữ liệu cho huấn luyện và kiểm định;
(iii) cả hai lớp được kết hợp bằng lớp `Trainer`, cho phép ta
huấn luyện mô hình trên nhiều nền tảng phần cứng khác nhau.
Hầu hết code trong cuốn sách này điều chỉnh `Module` và `DataModule`. Ta sẽ đề cập đến lớp `Trainer` chỉ khi ta thảo luận về GPU, CPU, huấn luyện song song và các thuật toán tối ưu hóa.


```python
import time
import numpy as np
from d2l import torch as d2l
import torch
from torch import nn
```


## Tiện ích
<a id="oo-design-utilities"></a>

Ta cần một vài tiện ích để đơn giản hóa lập trình hướng đối tượng trong các Jupyter notebook. Một trong những thách thức là các định nghĩa lớp có xu hướng là các khối code khá dài. Tính dễ đọc của notebook đòi hỏi các đoạn code ngắn, xen kẽ với các giải thích, một yêu cầu không tương thích với phong cách lập trình phổ biến cho các thư viện Python. Hàm tiện ích đầu tiên cho phép ta đăng ký các hàm như là các phương thức trong một lớp *sau khi* lớp đã được tạo. Thực ra, ta thậm chí có thể làm như vậy *ngay cả sau khi* ta đã tạo các thực thể của lớp! Nó cho phép ta chia việc cài đặt của một lớp thành nhiều khối code.

```python
def add_to_class(Class):  
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
```

Hãy xem nhanh cách sử dụng nó. Ta dự định cài đặt một lớp `A` với một phương thức `do`. Thay vì có code cho cả `A` và `do` trong cùng một khối code, ta có thể trước tiên khai báo lớp `A` và tạo một thực thể `a`.

```python
class A:
    def __init__(self):
        self.b = 1

a = A()
```

Tiếp theo ta định nghĩa phương thức `do` như thường lệ, nhưng không nằm trong phạm vi lớp `A`. Thay vào đó, ta trang trí phương thức này bằng `add_to_class` với lớp `A` là đối số. Khi làm như vậy, phương thức có thể truy cập các biến thành viên của `A` giống như ta mong đợi nếu nó được đưa vào trong định nghĩa của `A`. Hãy xem điều gì xảy ra khi ta gọi nó cho thực thể `a`.

```python
@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)

a.do()
```

Tiện ích thứ hai là một lớp tiện ích lưu tất cả các đối số trong phương thức `__init__` của một lớp như là các thuộc tính lớp. Điều này cho phép ta mở rộng chữ ký gọi hàm khởi tạo một cách ẩn mà không cần code bổ sung.

```python
class HyperParameters:  
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
```

Ta trì hoãn việc cài đặt vào [sec_utils](#sec_utils). Để sử dụng nó, ta định nghĩa lớp kế thừa từ `HyperParameters` và gọi `save_hyperparameters` trong phương thức `__init__`.

```python
# Call the fully implemented HyperParameters class saved in d2l
class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)
```

Tiện ích cuối cùng cho phép ta vẽ đồ thị tiến trình thí nghiệm một cách tương tác trong khi nó đang diễn ra. Để kính trọng [TensorBoard](https://www.tensorflow.org/tensorboard) mạnh mẽ hơn nhiều (và phức tạp hơn), ta đặt tên nó là `ProgressBoard`. Việc cài đặt được trì hoãn vào [sec_utils](#sec_utils). Hiện tại, hãy đơn giản xem nó hoạt động.

Phương thức `draw` vẽ một điểm `(x, y)` trong hình, với `label` được chỉ định trong chú giải. Tham số tùy chọn `every_n` làm mịn đường bằng cách chỉ hiển thị $1/n$ điểm trong hình. Các giá trị của chúng được tính trung bình từ $n$ điểm hàng xóm trong hình ban đầu.

```python
class ProgressBoard(d2l.HyperParameters):  
    """The board that plots data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented
```

Trong ví dụ sau, ta vẽ `sin` và `cos` với độ mịn khác nhau. Nếu bạn chạy khối code này, bạn sẽ thấy các đường phát triển theo hoạt ảnh.

```python
board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)
```

## Mô hình
<a id="subsec_oo-design-models"></a>

Lớp `Module` là lớp cơ sở của tất cả các mô hình ta sẽ cài đặt. Ít nhất ta cần ba phương thức. Phương thức đầu tiên, `__init__`, lưu các tham số có thể học, phương thức `training_step` chấp nhận một batch dữ liệu để trả về giá trị mất mát, và cuối cùng, `configure_optimizers` trả về phương thức tối ưu hóa, hoặc danh sách chúng, được dùng để cập nhật các tham số có thể học. Tùy chọn ta có thể định nghĩa `validation_step` để báo cáo các thước đo đánh giá.
Đôi khi ta đưa code tính đầu ra vào một phương thức `forward` riêng để làm cho nó có thể tái sử dụng hơn.


```python
class Module(d2l.nn_Module, d2l.HyperParameters):  
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```


Bạn có thể nhận thấy rằng `Module` là một lớp con của `nn.Module`, lớp cơ sở của mạng nơ-ron trong PyTorch.
Nó cung cấp các tính năng tiện lợi để xử lý các mạng nơ-ron. Chẳng hạn, nếu ta định nghĩa một phương thức `forward`, chẳng hạn `forward(self, X)`, thì với một thực thể `a` ta có thể gọi phương thức này bằng `a(X)`. Điều này hoạt động vì nó gọi phương thức `forward` trong phương thức `__call__` tích hợp. Bạn có thể tìm thêm chi tiết và ví dụ về `nn.Module` trong [sec_model_construction](#sec_model_construction).


##  Dữ liệu
<a id="oo-design-data"></a>

Lớp `DataModule` là lớp cơ sở cho dữ liệu. Khá thường xuyên phương thức `__init__` được dùng để chuẩn bị dữ liệu. Điều này bao gồm tải xuống và tiền xử lý nếu cần. `train_dataloader` trả về bộ tải dữ liệu cho tập dữ liệu huấn luyện. Bộ tải dữ liệu là một bộ tạo (Python) tạo ra một batch dữ liệu mỗi khi được sử dụng. Batch này sau đó được đưa vào phương thức `training_step` của `Module` để tính hàm mất mát. Có một `val_dataloader` tùy chọn để trả về bộ tải tập dữ liệu kiểm định. Nó hoạt động theo cách tương tự, ngoại trừ nó tạo ra các batch dữ liệu cho phương thức `validation_step` trong `Module`.

```python
class DataModule(d2l.HyperParameters):  
    """The base class of data."""
    if tab.selected('mxnet', 'pytorch'):
        def __init__(self, root='../data', num_workers=4):
            self.save_hyperparameters()

    if tab.selected('tensorflow', 'jax'):
        def __init__(self, root='../data'):
            self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
```

## Huấn luyện
<a id="oo-design-training"></a>


```python
class Trainer(d2l.HyperParameters):  
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        def fit(self, model, data):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    if tab.selected('jax'):
        def fit(self, model, data, key=None):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()

            if key is None:
                root_key = d2l.get_key()
            else:
                root_key = key
            params_key, dropout_key = jax.random.split(root_key)
            key = {'params': params_key, 'dropout': dropout_key}

            dummy_input = next(iter(self.train_dataloader))[:-1]
            variables = model.apply_init(dummy_input, key=key)
            params = variables['params']

            if 'batch_stats' in variables.keys():
                # Here batch_stats will be used later (e.g., for batch norm)
                batch_stats = variables['batch_stats']
            else:
                batch_stats = {}

            # Flax uses optax under the hood for a single state obj TrainState.
            # More will be discussed later in the dropout and batch
            # normalization section
            class TrainState(train_state.TrainState):
                batch_stats: Any
                dropout_rng: jax.random.PRNGKeyArray

            self.state = TrainState.create(apply_fn=model.apply,
                                           params=params,
                                           batch_stats=batch_stats,
                                           dropout_rng=dropout_key,
                                           tx=model.configure_optimizers())
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
```

## Tóm tắt

Để làm nổi bật thiết kế hướng đối tượng
cho cài đặt deep learning tương lai của ta,
các lớp trên chỉ đơn giản là cho thấy cách các đối tượng của chúng
lưu trữ dữ liệu và tương tác với nhau.
Ta sẽ tiếp tục làm phong phú các cài đặt của các lớp này,
chẳng hạn thông qua `@add_to_class`,
trong phần còn lại của cuốn sách.
Hơn nữa,
các lớp đã được cài đặt đầy đủ này
được lưu trong [thư viện D2L](https://github.com/d2l-ai/d2l-en/tree/master/d2l),
một *bộ công cụ nhẹ* giúp mô hình hóa có cấu trúc cho deep learning trở nên dễ dàng.
Cụ thể, nó tạo điều kiện tái sử dụng nhiều thành phần giữa các dự án mà không thay đổi nhiều. Chẳng hạn, ta có thể thay thế chỉ bộ tối ưu hóa, chỉ mô hình, chỉ tập dữ liệu, v.v.;
mức độ mô đun này mang lại lợi ích xuyên suốt cuốn sách về mặt ngắn gọn và đơn giản (đó là lý do ta thêm nó) và nó có thể làm điều tương tự cho các dự án của riêng bạn.


## Bài tập

1. Tìm các cài đặt đầy đủ của các lớp trên được lưu trong [thư viện D2L](https://github.com/d2l-ai/d2l-en/tree/master/d2l). Ta khuyến khích bạn xem xét chi tiết cài đặt một khi bạn đã quen hơn với mô hình deep learning.
1. Xóa câu lệnh `save_hyperparameters` trong lớp `B`. Bạn có vẫn có thể in `self.a` và `self.b` không? Tùy chọn: nếu bạn đã đi sâu vào cài đặt đầy đủ của lớp `HyperParameters`, bạn có thể giải thích tại sao không?


[Thảo luận](https://discuss.d2l.ai/t/6646)
