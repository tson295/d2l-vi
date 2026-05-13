# Lập Lịch Tốc Độ Học
<a id="sec_scheduler"></a>

Cho đến nay, chúng ta chủ yếu tập trung vào các *thuật toán* tối ưu hóa để cập nhật các vector trọng số, thay vì *tốc độ* mà chúng được cập nhật. Tuy nhiên, điều chỉnh tốc độ học thường cũng quan trọng không kém thuật toán thực tế. Có một số khía cạnh cần xem xét:

* Rõ ràng nhất là *độ lớn* của tốc độ học rất quan trọng. Nếu quá lớn, tối ưu hóa phân kỳ; nếu quá nhỏ, huấn luyện mất quá lâu hoặc chúng ta thu được kết quả dưới tối ưu. Trước đây chúng ta đã thấy rằng số điều kiện của bài toán có vai trò quan trọng (xem ví dụ [sec_momentum](#sec_momentum) để biết chi tiết). Về trực giác, đó là tỷ số giữa lượng thay đổi theo hướng ít nhạy nhất và hướng nhạy nhất.
* Thứ hai, tốc độ suy giảm cũng quan trọng không kém. Nếu tốc độ học vẫn lớn, chúng ta có thể chỉ dao động quanh cực tiểu và do đó không đạt nghiệm tối ưu. [sec_minibatch_sgd](#sec_minibatch_sgd) đã thảo luận điều này khá chi tiết và chúng ta đã phân tích các đảm bảo hiệu năng trong [sec_sgd](#sec_sgd). Tóm lại, chúng ta muốn tốc độ học suy giảm, nhưng có lẽ chậm hơn $\mathcal{O}(t^{-\frac{1}{2}})$, vốn là một lựa chọn tốt cho các bài toán lồi.
* Một khía cạnh khác quan trọng không kém là *khởi tạo*. Điều này liên quan cả đến cách các tham số được đặt ban đầu (xem lại [sec_numerical_stability](#sec_numerical_stability) để biết chi tiết) và cách chúng tiến hóa ban đầu. Điều này được gọi bằng tên *warmup*, tức là ban đầu chúng ta bắt đầu di chuyển về phía nghiệm nhanh đến mức nào. Các bước lớn lúc đầu có thể không có lợi, đặc biệt vì tập tham số ban đầu là ngẫu nhiên. Các hướng cập nhật ban đầu cũng có thể khá vô nghĩa.
* Cuối cùng, có một số biến thể tối ưu hóa thực hiện điều chỉnh tốc độ học theo chu kỳ. Điều này nằm ngoài phạm vi của chương hiện tại. Chúng tôi khuyến nghị độc giả xem chi tiết trong Izmailov.Podoprikhin.Garipov.ea.2018, ví dụ cách thu được nghiệm tốt hơn bằng cách lấy trung bình trên toàn bộ một *đường đi* của tham số.

Do cần rất nhiều chi tiết để quản lý tốc độ học, hầu hết các framework deep learning đều có công cụ để xử lý việc này tự động. Trong chương hiện tại, chúng ta sẽ xem xét các tác động mà các lịch khác nhau gây ra lên độ chính xác, đồng thời chỉ ra cách quản lý điều này hiệu quả thông qua một *bộ lập lịch tốc độ học*.

## Bài Toán Đồ Chơi

Chúng ta bắt đầu với một bài toán đồ chơi đủ rẻ để tính toán dễ dàng, nhưng vẫn đủ không tầm thường để minh họa một số khía cạnh chính. Để làm điều đó, chúng ta chọn một phiên bản hơi hiện đại hóa của LeNet (kích hoạt `relu` thay vì `sigmoid`, MaxPooling thay vì AveragePooling), áp dụng cho Fashion-MNIST. Hơn nữa, chúng ta hybridize mạng để tăng hiệu năng. Vì phần lớn mã là chuẩn, chúng ta chỉ giới thiệu các điểm cơ bản mà không thảo luận chi tiết thêm. Xem [chap_cnn](#chap_cnn) để ôn lại khi cần.

```python
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, lr_scheduler, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.HybridSequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device):
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```python
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.optim import lr_scheduler

def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
          scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```python
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu(), custom_callback = False):
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    if custom_callback is False:
        net.fit(train_iter, epochs=num_epochs, verbose=0,
                callbacks=[callback])
    else:
         net.fit(train_iter, epochs=num_epochs, verbose=0,
                 callbacks=[callback, custom_callback])
    return net
```

Hãy xem điều gì xảy ra nếu chúng ta gọi thuật toán này với các thiết lập mặc định, chẳng hạn tốc độ học $0.3$ và huấn luyện trong $30$ vòng lặp. Lưu ý rằng độ chính xác huấn luyện tiếp tục tăng, trong khi tiến triển về độ chính xác kiểm tra bị đình trệ sau một điểm. Khoảng cách giữa hai đường cong cho thấy hiện tượng quá khớp.

```python
#@tab mxnet
lr, num_epochs = 0.3, 30
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```python
#@tab pytorch
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```python
#@tab tensorflow
lr, num_epochs = 0.3, 30
train(net, train_iter, test_iter, num_epochs, lr)
```

## Bộ Lập Lịch

Một cách điều chỉnh tốc độ học là đặt nó tường minh ở mỗi bước. Điều này được thực hiện thuận tiện bằng phương thức `set_learning_rate`. Chúng ta có thể điều chỉnh giảm sau mỗi epoch (hoặc thậm chí sau mỗi minibatch), ví dụ theo cách động để phản hồi tiến triển của tối ưu hóa.

```python
#@tab mxnet
trainer.set_learning_rate(0.1)
print(f'learning rate is now {trainer.learning_rate:.2f}')
```

```python
#@tab pytorch
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')
```

```python
#@tab tensorflow
lr = 0.1
dummy_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
dummy_model.compile(tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
print(f'learning rate is now ,', dummy_model.optimizer.lr.numpy())
```

Tổng quát hơn, chúng ta muốn định nghĩa một bộ lập lịch. Khi được gọi với số cập nhật, nó trả về giá trị tốc độ học phù hợp. Hãy định nghĩa một bộ đơn giản đặt tốc độ học thành $\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$.

```python
#@tab all
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
```

Hãy vẽ hành vi của nó trên một khoảng giá trị.

```python
#@tab all
scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Bây giờ hãy xem điều này diễn ra thế nào khi huấn luyện trên Fashion-MNIST. Chúng ta chỉ cần cung cấp bộ lập lịch như một đối số bổ sung cho thuật toán huấn luyện.

```python
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```python
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```python
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

Điều này hoạt động tốt hơn khá nhiều so với trước. Hai điều nổi bật: đường cong trơn hơn đáng kể so với trước. Thứ hai, có ít quá khớp hơn. Đáng tiếc là chưa có câu trả lời rõ ràng về mặt *lý thuyết* cho việc vì sao một số chiến lược nhất định dẫn đến ít quá khớp hơn. Có lập luận rằng kích thước bước nhỏ hơn sẽ dẫn đến các tham số gần 0 hơn và do đó đơn giản hơn. Tuy nhiên, điều này không giải thích hoàn toàn hiện tượng vì chúng ta không thực sự dừng sớm mà chỉ giảm tốc độ học một cách nhẹ nhàng.

## Chính Sách

Mặc dù không thể bao quát toàn bộ sự đa dạng của các bộ lập lịch tốc độ học, chúng ta cố gắng đưa ra một tổng quan ngắn về các chính sách phổ biến bên dưới. Các lựa chọn phổ biến là suy giảm đa thức và các lịch hằng từng đoạn. Ngoài ra, các lịch tốc độ học cosine đã được phát hiện là hoạt động tốt theo kinh nghiệm trên một số bài toán. Cuối cùng, trên một số bài toán, việc warm up bộ tối ưu hóa trước khi dùng tốc độ học lớn là có lợi.

### Bộ Lập Lịch Theo Hệ Số

Một phương án thay thế cho suy giảm đa thức là suy giảm nhân, tức là $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ với $\alpha \in (0, 1)$. Để ngăn tốc độ học suy giảm vượt quá một cận dưới hợp lý, phương trình cập nhật thường được sửa thành $\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$.

```python
#@tab all
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(d2l.arange(50), [scheduler(t) for t in range(50)])
```

Điều này cũng có thể được thực hiện bằng một bộ lập lịch tích hợp trong MXNet thông qua đối tượng `lr_scheduler.FactorScheduler`. Nó nhận thêm một vài tham số, chẳng hạn giai đoạn warmup, chế độ warmup (tuyến tính hoặc hằng), số cập nhật mong muốn tối đa, v.v.; từ đây trở đi, chúng ta sẽ dùng các bộ lập lịch tích hợp khi phù hợp và chỉ giải thích chức năng của chúng ở đây. Như đã minh họa, việc xây dựng bộ lập lịch riêng khi cần khá đơn giản.

### Bộ Lập Lịch Đa Hệ Số

Một chiến lược phổ biến để huấn luyện các mạng sâu là giữ tốc độ học hằng từng đoạn và thỉnh thoảng giảm nó theo một lượng cho trước. Tức là, cho một tập các thời điểm cần giảm tốc độ, chẳng hạn $s = \{5, 10, 20\}$, giảm $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ bất cứ khi nào $t \in s$. Giả sử các giá trị được giảm một nửa ở mỗi bước, chúng ta có thể cài đặt điều này như sau.

```python
#@tab mxnet
scheduler = lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5,
                                              base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```python
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(d2l.arange(num_epochs), [get_lr(trainer, scheduler)
                                  for t in range(num_epochs)])
```

```python
#@tab tensorflow
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr

    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Trực giác đằng sau lịch tốc độ học hằng từng đoạn này là ta để tối ưu hóa tiếp diễn cho đến khi đạt một điểm dừng xét theo phân phối của các vector trọng số. Sau đó (và chỉ sau đó), ta mới giảm tốc độ để thu được một đại diện chất lượng cao hơn cho một cực tiểu cục bộ tốt. Ví dụ dưới đây cho thấy cách điều này có thể tạo ra các nghiệm tốt hơn một chút.

```python
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```python
#@tab pytorch
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```python
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### Bộ Lập Lịch Cosine

Một heuristic khá khó hiểu được đề xuất bởi Loshchilov.Hutter.2016. Nó dựa trên quan sát rằng chúng ta có thể không muốn giảm tốc độ học quá mạnh lúc đầu và hơn nữa, có thể muốn "tinh chỉnh" nghiệm ở cuối bằng một tốc độ học rất nhỏ. Điều này dẫn đến một lịch dạng cosine với dạng hàm sau cho tốc độ học trong khoảng $t \in [0, T]$.

$$\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2} \left(1 + \cos(\pi t/T)\right)$$


Ở đây $\eta_0$ là tốc độ học ban đầu, $\eta_T$ là tốc độ mục tiêu tại thời điểm $T$. Hơn nữa, với $t > T$, chúng ta đơn giản ghim giá trị tại $\eta_T$ mà không tăng lại. Trong ví dụ sau, chúng ta đặt bước cập nhật tối đa $T = 20$.

```python
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(max_update=20, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```python
#@tab pytorch, tensorflow
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Trong bối cảnh thị giác máy tính, lịch này *có thể* dẫn đến kết quả cải thiện. Tuy nhiên, lưu ý rằng các cải thiện như vậy không được đảm bảo (như có thể thấy bên dưới).

```python
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```python
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```python
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### Warmup

Trong một số trường hợp, khởi tạo tham số không đủ để đảm bảo một nghiệm tốt. Đây đặc biệt là vấn đề đối với một số thiết kế mạng nâng cao có thể dẫn đến các bài toán tối ưu hóa không ổn định. Chúng ta có thể xử lý điều này bằng cách chọn tốc độ học đủ nhỏ để ngăn phân kỳ lúc đầu. Đáng tiếc là điều này có nghĩa tiến triển chậm. Ngược lại, tốc độ học lớn ban đầu dẫn đến phân kỳ.

Một cách sửa khá đơn giản cho thế lưỡng nan này là dùng một giai đoạn warmup trong đó tốc độ học *tăng* lên cực đại ban đầu, rồi làm nguội tốc độ cho đến cuối quá trình tối ưu hóa. Để đơn giản, người ta thường dùng tăng tuyến tính cho mục đích này. Điều này dẫn đến một lịch có dạng được chỉ ra bên dưới.

```python
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(20, warmup_steps=5, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```python
#@tab pytorch, tensorflow
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Lưu ý rằng mạng hội tụ tốt hơn lúc đầu (đặc biệt hãy quan sát hiệu năng trong 5 epoch đầu tiên).

```python
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```python
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```python
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

Warmup có thể được áp dụng cho bất kỳ bộ lập lịch nào (không chỉ cosine). Để thảo luận chi tiết hơn về các lịch tốc độ học và nhiều thí nghiệm hơn, xem thêm [Gotmare.Keskar.Xiong.ea.2018]. Cụ thể, họ phát hiện rằng giai đoạn warmup giới hạn mức độ phân kỳ của tham số trong các mạng rất sâu. Điều này hợp lý về mặt trực giác vì chúng ta kỳ vọng phân kỳ đáng kể do khởi tạo ngẫu nhiên ở những phần của mạng mất nhiều thời gian nhất để tiến triển lúc đầu.

## Tóm Tắt

* Giảm tốc độ học trong quá trình huấn luyện có thể dẫn đến độ chính xác được cải thiện và (khó hiểu nhất là) giảm quá khớp của mô hình.
* Giảm tốc độ học theo từng đoạn mỗi khi tiến triển đã chững lại là hiệu quả trong thực tế. Về cơ bản, điều này đảm bảo chúng ta hội tụ hiệu quả đến một nghiệm phù hợp và chỉ khi đó mới giảm phương sai vốn có của các tham số bằng cách giảm tốc độ học.
* Các bộ lập lịch cosine phổ biến cho một số bài toán thị giác máy tính. Xem ví dụ [GluonCV](http://gluon-cv.mxnet.io) để biết chi tiết về một bộ lập lịch như vậy.
* Một giai đoạn warmup trước tối ưu hóa có thể ngăn phân kỳ.
* Tối ưu hóa phục vụ nhiều mục đích trong deep learning. Ngoài việc tối thiểu hóa mục tiêu huấn luyện, các lựa chọn khác nhau về thuật toán tối ưu hóa và lập lịch tốc độ học có thể dẫn đến mức độ khái quát hóa và quá khớp khá khác nhau trên tập kiểm tra (với cùng lượng lỗi huấn luyện).

## Bài Tập

1. Thử nghiệm với hành vi tối ưu hóa cho một tốc độ học cố định cho trước. Mô hình tốt nhất bạn có thể thu được theo cách này là gì?
1. Hội tụ thay đổi như thế nào nếu bạn thay đổi số mũ của mức giảm trong tốc độ học? Hãy dùng `PolyScheduler` để thuận tiện trong các thí nghiệm.
1. Áp dụng bộ lập lịch cosine cho các bài toán thị giác máy tính lớn, ví dụ huấn luyện ImageNet. Nó ảnh hưởng thế nào đến hiệu năng so với các bộ lập lịch khác?
1. Warmup nên kéo dài bao lâu?
1. Bạn có thể kết nối tối ưu hóa và lấy mẫu không? Hãy bắt đầu bằng cách dùng các kết quả từ Welling.Teh.2011 về Stochastic Gradient Langevin Dynamics.


[Discussions](https://discuss.d2l.ai/t/1080)
