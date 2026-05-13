# Song Song Tự Động
<a id="sec_auto_para"></a>


Các framework deep learning (ví dụ MXNet và PyTorch) tự động xây dựng đồ thị tính toán ở backend. Dùng một
đồ thị tính toán, hệ thống biết tất cả các phụ thuộc,
và có thể chọn lọc thực thi nhiều tác vụ không phụ thuộc lẫn nhau song song để
cải thiện tốc độ. Chẳng hạn, [fig_asyncgraph](#fig_asyncgraph) trong [sec_async](#sec_async) khởi tạo hai biến độc lập. Do đó, hệ thống có thể chọn thực thi chúng song song.


Thông thường, một toán tử đơn lẻ sẽ dùng toàn bộ tài nguyên tính toán trên mọi CPU hoặc trên một GPU đơn. Ví dụ, toán tử `dot` sẽ dùng tất cả lõi (và luồng) trên tất cả CPU, ngay cả khi có nhiều bộ xử lý CPU trên một máy. Điều tương tự áp dụng cho một GPU đơn. Vì vậy, song song hóa không quá hữu ích với các máy tính một thiết bị. Với nhiều thiết bị, vấn đề này quan trọng hơn. Mặc dù song song hóa thường liên quan nhất giữa nhiều GPU, thêm CPU cục bộ sẽ tăng hiệu năng một chút. Ví dụ, xem Hadjis.Zhang.Mitliagkas.ea.2016, tập trung vào huấn luyện các mô hình thị giác máy tính kết hợp GPU và CPU. Với sự tiện lợi của một framework tự động song song hóa, chúng ta có thể đạt cùng mục tiêu chỉ trong vài dòng mã Python. Rộng hơn, thảo luận của chúng ta về tính toán song song tự động tập trung vào tính toán song song dùng cả CPU và GPU, cũng như song song hóa tính toán và giao tiếp.

Lưu ý rằng chúng ta cần ít nhất hai GPU để chạy các thí nghiệm trong phần này.

```python
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```python
#@tab pytorch
from d2l import torch as d2l
import torch
```

## Tính Toán Song Song Trên GPU

Hãy bắt đầu bằng cách định nghĩa một workload tham chiếu để kiểm thử: hàm `run` bên dưới thực hiện 10 phép nhân ma trận-ma trận trên thiết bị chúng ta chọn, dùng dữ liệu được cấp phát vào hai biến: `x_gpu1` và `x_gpu2`.

```python
#@tab mxnet
devices = d2l.try_all_gpus()
def run(x):
    return [x.dot(x) for _ in range(50)]

x_gpu1 = np.random.uniform(size=(4000, 4000), ctx=devices[0])
x_gpu2 = np.random.uniform(size=(4000, 4000), ctx=devices[1])
```

```python
#@tab pytorch
devices = d2l.try_all_gpus()
def run(x):
    return [x.mm(x) for _ in range(50)]

x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])
```


Bây giờ chúng ta áp dụng hàm cho dữ liệu. Để đảm bảo cache không ảnh hưởng đến kết quả, chúng ta warm up các thiết bị bằng cách thực hiện một lượt đơn trên từng thiết bị trước khi đo. `torch.cuda.synchronize()` chờ tất cả kernel trong mọi stream trên một thiết bị CUDA hoàn tất. Nó nhận một đối số `device`, là thiết bị mà chúng ta cần đồng bộ. Nó dùng thiết bị hiện tại, được cho bởi `current_device()`, nếu đối số device là `None` (mặc định).

```python
#@tab mxnet
run(x_gpu1)  # Warm-up both devices
run(x_gpu2)
npx.waitall()

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    npx.waitall()
```

```python
#@tab pytorch
run(x_gpu1)
run(x_gpu2)  # Warm-up all devices
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])
```


Nếu loại bỏ câu lệnh `synchronize` giữa hai tác vụ, hệ thống được tự do song song hóa tính toán trên cả hai thiết bị một cách tự động.

```python
#@tab mxnet
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    npx.waitall()
```

```python
#@tab pytorch
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()
```

Trong trường hợp trên, tổng thời gian thực thi nhỏ hơn tổng các phần riêng lẻ, vì framework deep learning tự động lập lịch tính toán trên cả hai thiết bị GPU mà không cần người dùng viết mã tinh vi.


## Tính Toán và Giao Tiếp Song Song

Trong nhiều trường hợp, chúng ta cần di chuyển dữ liệu giữa các thiết bị khác nhau, chẳng hạn giữa CPU và GPU, hoặc giữa các GPU khác nhau.
Chẳng hạn,
điều này xảy ra khi chúng ta muốn thực hiện tối ưu hóa phân tán, nơi cần tổng hợp gradient trên nhiều card tăng tốc. Hãy mô phỏng điều này bằng cách tính trên GPU rồi sao chép kết quả trở lại CPU.

```python
#@tab mxnet
def copy_to_cpu(x):
    return [y.copyto(npx.cpu()) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```python
#@tab pytorch
def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    torch.cuda.synchronize()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()
```


Điều này hơi kém hiệu quả. Lưu ý rằng chúng ta đã có thể bắt đầu sao chép các phần của `y` về CPU trong khi phần còn lại của danh sách vẫn đang được tính. Tình huống này xảy ra, ví dụ, khi chúng ta tính gradient (backprop) trên một minibatch. Gradient của một số tham số sẽ sẵn sàng sớm hơn của các tham số khác. Do đó, bắt đầu dùng băng thông bus PCI-Express trong khi GPU vẫn đang chạy là có lợi. Trong PyTorch, một số hàm như `to()` và `copy_()` chấp nhận đối số tường minh `non_blocking`, cho phép bên gọi bỏ qua đồng bộ hóa khi không cần thiết. Đặt `non_blocking=True` cho phép chúng ta mô phỏng kịch bản này.

```python
#@tab mxnet
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```python
#@tab pytorch
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, True)
    torch.cuda.synchronize()
```

Tổng thời gian cần cho cả hai thao tác (như kỳ vọng) nhỏ hơn tổng các phần riêng lẻ.
Lưu ý rằng tác vụ này khác với tính toán song song vì nó dùng một tài nguyên khác: bus giữa CPU và GPU. Trên thực tế, chúng ta có thể tính toán trên cả hai thiết bị và giao tiếp, tất cả cùng lúc. Như đã lưu ý ở trên, có một phụ thuộc giữa tính toán và giao tiếp: `y[i]` phải được tính trước khi có thể được sao chép về CPU. May mắn là hệ thống có thể sao chép `y[i-1]` trong khi tính `y[i]` để giảm tổng thời gian chạy.

Chúng ta kết thúc bằng một minh họa về đồ thị tính toán và các phụ thuộc của nó cho một MLP hai lớp đơn giản khi huấn luyện trên một CPU và hai GPU, như mô tả trong [fig_twogpu](#fig_twogpu). Việc lập lịch thủ công chương trình song song thu được từ đây sẽ khá vất vả. Đây là nơi backend tính toán dựa trên đồ thị trở nên có lợi cho tối ưu hóa.

![Đồ thị tính toán và các phụ thuộc của một MLP hai lớp trên một CPU và hai GPU.](../img/twogpu.svg)
<a id="fig_twogpu"></a>


## Tóm Tắt

* Các hệ thống hiện đại có nhiều loại thiết bị, chẳng hạn nhiều GPU và CPU. Chúng có thể được dùng song song, bất đồng bộ.
* Các hệ thống hiện đại cũng có nhiều loại tài nguyên cho giao tiếp, chẳng hạn PCI Express, lưu trữ (thường là ổ thể rắn hoặc qua mạng) và băng thông mạng. Chúng có thể được dùng song song để đạt hiệu quả đỉnh.
* Backend có thể cải thiện hiệu năng thông qua tính toán và giao tiếp song song tự động.

## Bài Tập

1. Tám thao tác đã được thực hiện trong hàm `run` được định nghĩa trong phần này. Không có phụ thuộc giữa chúng. Hãy thiết kế một thí nghiệm để xem liệu framework deep learning có tự động thực thi chúng song song hay không.
1. Khi workload của một toán tử riêng lẻ đủ nhỏ, song song hóa có thể hữu ích ngay cả trên một CPU hoặc GPU đơn. Hãy thiết kế một thí nghiệm để xác minh điều này.
1. Thiết kế một thí nghiệm dùng tính toán song song trên CPU, GPU và giao tiếp giữa hai loại thiết bị.
1. Dùng một trình gỡ lỗi như [Nsight](https://developer.nvidia.com/nsight-compute-2019_5) của NVIDIA để xác minh rằng mã của bạn hiệu quả.
1. Thiết kế các tác vụ tính toán bao gồm các phụ thuộc dữ liệu phức tạp hơn, rồi chạy thí nghiệm để xem liệu bạn có thể thu được kết quả đúng trong khi cải thiện hiệu năng hay không.


[Discussions](https://discuss.d2l.ai/t/1681)
