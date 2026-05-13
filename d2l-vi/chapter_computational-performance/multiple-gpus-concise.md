# Cài Đặt Ngắn Gọn Cho Nhiều GPU
<a id="sec_multi_gpu_concise"></a>

Cài đặt song song hóa từ đầu cho mọi mô hình mới không hề thú vị. Hơn nữa, việc tối ưu các công cụ đồng bộ hóa để đạt hiệu năng cao đem lại lợi ích đáng kể. Trong phần sau, chúng ta sẽ chỉ ra cách làm điều này bằng các API cấp cao của các framework deep learning.
Toán học và thuật toán giống như trong [sec_multi_gpu](#sec_multi_gpu).
Không có gì ngạc nhiên khi bạn sẽ cần ít nhất hai GPU để chạy mã trong phần này.

```python
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```python
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## [**Một Mạng Đồ Chơi**]

Hãy dùng một mạng có ý nghĩa hơn một chút so với LeNet trong [sec_multi_gpu](#sec_multi_gpu), nhưng vẫn đủ dễ và nhanh để huấn luyện.
Chúng ta chọn một biến thể ResNet-18 [He.Zhang.Ren.ea.2016]. Vì ảnh đầu vào rất nhỏ, chúng ta sửa nó một chút. Cụ thể, khác biệt so với [sec_resnet](#sec_resnet) là chúng ta dùng kernel tích chập, stride và padding nhỏ hơn ở đầu.
Hơn nữa, chúng ta loại bỏ lớp max-pooling.

```python
#@tab mxnet
def resnet18(num_classes):
    """A slightly modified ResNet-18 model."""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the max-pooling layer
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

```python
#@tab pytorch
def resnet18(num_classes, in_channels=1):
    """A slightly modified ResNet-18 model."""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(out_channels, use_1x1conv=True, 
                                        strides=2))
            else:
                blk.append(d2l.Residual(out_channels))
        return nn.Sequential(*blk)

    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the max-pooling layer
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

## Khởi Tạo Mạng


Chúng ta sẽ khởi tạo mạng bên trong vòng lặp huấn luyện.
Để ôn lại các phương pháp khởi tạo, xem [sec_numerical_stability](#sec_numerical_stability).

```python
#@tab mxnet
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# Initialize all the parameters of the network
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

```python
#@tab pytorch
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# We will initialize the network inside the training loop
```


```python
#@tab mxnet
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```


```python
#@tab mxnet
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(devices[0])[0], weight.data(devices[1])[0]
```


```python
#@tab mxnet
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    """Compute the accuracy for a model on a dataset using multiple GPUs."""
    # Query the list of devices
    devices = list(net.collect_params().values())[0].list_ctx()
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        # Run in parallel
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```

## [**Huấn Luyện**]

Như trước, mã huấn luyện cần thực hiện một số chức năng cơ bản để song song hóa hiệu quả:

* Tham số mạng cần được khởi tạo trên tất cả thiết bị.
* Khi lặp qua tập dữ liệu, các minibatch cần được chia trên tất cả thiết bị.
* Chúng ta tính mất mát và gradient của nó song song trên các thiết bị.
* Gradient được tổng hợp và tham số được cập nhật tương ứng.

Cuối cùng, chúng ta tính độ chính xác (lại song song) để báo cáo hiệu năng cuối cùng của mạng. Quy trình huấn luyện khá giống các cài đặt trong các chương trước, ngoại trừ việc chúng ta cần chia và tổng hợp dữ liệu.

```python
#@tab mxnet
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [d2l.try_gpu(i) for i in range(num_gpus)]
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for features, labels in train_iter:
            X_shards, y_shards = d2l.split_batch(features, labels, ctx)
            with autograd.record():
                ls = [loss(net(X_shard), y_shard) for X_shard, y_shard
                      in zip(X_shards, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        npx.waitall()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpus(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(ctx)}')
```

```python
#@tab pytorch
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(module):
        if type(module) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)
    # Set the model on multiple GPUs
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

Hãy xem điều này hoạt động thế nào trong thực tế. Để warm up, chúng ta [**huấn luyện mạng trên một GPU đơn.**]

```python
#@tab mxnet
train(num_gpus=1, batch_size=256, lr=0.1)
```

```python
#@tab pytorch
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

Tiếp theo, chúng ta [**dùng 2 GPU để huấn luyện**]. So với LeNet
được đánh giá trong [sec_multi_gpu](#sec_multi_gpu),
mô hình ResNet-18 phức tạp hơn đáng kể. Đây là nơi song song hóa thể hiện lợi thế. Thời gian tính toán lớn hơn có ý nghĩa so với thời gian đồng bộ hóa tham số. Điều này cải thiện khả năng mở rộng vì chi phí phụ cho song song hóa ít quan trọng hơn.

```python
#@tab mxnet
train(num_gpus=2, batch_size=512, lr=0.2)
```

```python
#@tab pytorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## Tóm Tắt

* Dữ liệu được tự động đánh giá trên các thiết bị nơi dữ liệu có mặt.
* Cẩn thận khởi tạo mạng trên từng thiết bị trước khi cố truy cập tham số trên thiết bị đó. Nếu không, bạn sẽ gặp lỗi.
* Các thuật toán tối ưu hóa tự động tổng hợp trên nhiều GPU.


## Bài Tập


1. Phần này dùng ResNet-18. Hãy thử các epoch, kích thước batch và tốc độ học khác nhau. Dùng nhiều GPU hơn cho tính toán. Điều gì xảy ra nếu bạn thử với 16 GPU (ví dụ trên một instance AWS p2.16xlarge)?
1. Đôi khi, các thiết bị khác nhau cung cấp năng lực tính toán khác nhau. Chúng ta có thể dùng GPU và CPU cùng lúc. Nên chia công việc như thế nào? Có đáng công không? Vì sao? Vì sao không?


[Discussions](https://discuss.d2l.ai/t/1403)
