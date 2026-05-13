# Mạng Nơ-ron Tích chập (LeNet)
<a id="sec_lenet"></a>

Bây giờ chúng ta có tất cả các thành phần cần thiết để lắp ráp
một CNN hoạt động đầy đủ.
Trong lần gặp gỡ trước với dữ liệu hình ảnh, chúng ta đã áp dụng
một mô hình tuyến tính với hồi quy softmax ([sec_softmax_scratch](#sec_softmax_scratch))
và một MLP ([sec_mlp-implementation](#sec_mlp-implementation))
cho các bức ảnh quần áo trong bộ dữ liệu Fashion-MNIST.
Để làm cho dữ liệu như vậy có thể xử lý được, chúng ta đã làm phẳng mỗi hình ảnh từ ma trận $28\times28$
thành vectơ có chiều dài cố định $784$ chiều,
và sau đó xử lý chúng trong các lớp kết nối đầy đủ.
Bây giờ chúng ta đã nắm được các lớp tích chập,
chúng ta có thể giữ lại cấu trúc không gian trong các hình ảnh.
Như một lợi ích thêm khi thay thế các lớp kết nối đầy đủ bằng các lớp tích chập,
chúng ta sẽ có các mô hình tiết kiệm hơn cần ít tham số hơn nhiều.

Trong phần này, chúng ta sẽ giới thiệu *LeNet*,
trong số các CNN được công bố đầu tiên
để thu hút sự chú ý rộng rãi về hiệu suất của nó trong các tác vụ thị giác máy tính.
Mô hình được giới thiệu bởi (và đặt theo tên) Yann LeCun,
khi đó là nhà nghiên cứu tại AT&T Bell Labs,
với mục đích nhận dạng các chữ số viết tay trong hình ảnh [LeCun.Bottou.Bengio.ea.1998].
Công trình này đại diện cho đỉnh cao
của một thập kỷ nghiên cứu phát triển công nghệ;
nhóm của LeCun đã công bố nghiên cứu đầu tiên thành công
huấn luyện CNN thông qua lan truyền ngược [LeCun.Boser.Denker.ea.1989].

Vào thời điểm đó LeNet đạt được kết quả xuất sắc
phù hợp với hiệu suất của máy vectơ hỗ trợ,
khi đó là cách tiếp cận thống trị trong học có giám sát, đạt tỷ lệ lỗi dưới 1% mỗi chữ số.
LeNet cuối cùng đã được điều chỉnh để nhận dạng chữ số
cho việc xử lý tiền gửi trong máy ATM.
Cho đến ngày nay, một số ATM vẫn chạy code
mà Yann LeCun và đồng nghiệp Leon Bottou đã viết vào những năm 1990!


```python
from d2l import torch as d2l
import torch
from torch import nn
```


## LeNet

Ở cấp độ cao, (**LeNet (LeNet-5) bao gồm hai phần:
(i) một bộ mã hóa tích chập bao gồm hai lớp tích chập; và
(ii) một khối dày đặc bao gồm ba lớp kết nối đầy đủ**).
Kiến trúc được tóm tắt trong [img_lenet](#img_lenet).

![Luồng dữ liệu trong LeNet. Đầu vào là một chữ số viết tay, đầu ra là xác suất qua 10 kết quả có thể.](../img/lenet.svg)
<a id="img_lenet"></a>

Các đơn vị cơ bản trong mỗi khối tích chập
là một lớp tích chập, hàm kích hoạt sigmoid,
và một phép toán gộp trung bình tiếp theo.
Lưu ý rằng trong khi ReLU và gộp tối đa hoạt động tốt hơn,
chúng vẫn chưa được phát hiện vào lúc đó.
Mỗi lớp tích chập sử dụng nhân $5\times 5$
và hàm kích hoạt sigmoid.
Các lớp này ánh xạ đầu vào được sắp xếp theo không gian
thành một số bản đồ đặc trưng hai chiều, thường
tăng số lượng kênh.
Lớp tích chập đầu tiên có 6 kênh đầu ra,
trong khi lớp thứ hai có 16.
Mỗi phép gộp $2\times2$ (sải bước 2)
giảm chiều đi một hệ số $4$ thông qua lấy mẫu thấp hơn không gian.
Khối tích chập phát ra đầu ra có hình dạng
(kích thước batch, số kênh, chiều cao, chiều rộng).

Để chuyển đầu ra từ khối tích chập
sang khối dày đặc,
chúng ta phải làm phẳng mỗi mẫu trong minibatch.
Nói cách khác, chúng ta lấy đầu vào bốn chiều này và chuyển đổi nó
thành đầu vào hai chiều được mong đợi bởi các lớp kết nối đầy đủ:
như một lời nhắc, biểu diễn hai chiều mà chúng ta mong muốn sử dụng chiều đầu tiên để lập chỉ số các mẫu trong minibatch
và chiều thứ hai để cung cấp biểu diễn vectơ phẳng của mỗi mẫu.
Khối dày đặc của LeNet có ba lớp kết nối đầy đủ,
với 120, 84, và 10 đầu ra, tương ứng.
Vì chúng ta vẫn đang thực hiện phân loại,
lớp đầu ra 10 chiều tương ứng
với số lớp đầu ra có thể.

Trong khi đến điểm mà bạn thực sự hiểu
những gì đang xảy ra bên trong LeNet có thể mất một chút công sức,
chúng ta hy vọng rằng đoạn code sau sẽ thuyết phục bạn
rằng việc triển khai các mô hình như vậy với các framework deep learning hiện đại
đơn giản đáng kể.
Chúng ta chỉ cần khởi tạo một khối `Sequential`
và nối các lớp thích hợp lại với nhau,
sử dụng khởi tạo Xavier như
được giới thiệu trong [subsec_xavier](#subsec_xavier).

```python
def init_cnn(module):  
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
```


Chúng ta đã có một chút tự do trong việc tái tạo LeNet ở chỗ chúng ta đã thay thế lớp kích hoạt Gaussian bằng
một lớp softmax. Điều này đơn giản hóa đáng kể việc triển khai, không kém vì
thực tế là bộ giải mã Gaussian hiếm khi được sử dụng ngày nay. Ngoài điều đó, mạng này khớp
với kiến trúc LeNet-5 gốc.


![Ký hiệu nén cho LeNet-5.](../img/lenet-vert.svg)
<a id="img_lenet_vert"></a>


Lưu ý rằng chiều cao và chiều rộng của biểu diễn
tại mỗi lớp trong toàn bộ khối tích chập
giảm dần (so với lớp trước).
Lớp tích chập đầu tiên sử dụng hai điểm ảnh đệm
để bù cho sự giảm chiều cao và chiều rộng
mà nếu không sẽ xảy ra khi sử dụng nhân $5 \times 5$.
Ngoài ra, kích thước hình ảnh $28 \times 28$ điểm ảnh trong bộ dữ liệu
MNIST OCR gốc là kết quả của việc *cắt bỏ* hai hàng điểm ảnh (và cột) từ
các bản quét gốc đo $32 \times 32$ điểm ảnh. Điều này được thực hiện chủ yếu để
tiết kiệm không gian (giảm 30%) vào thời điểm megabyte quan trọng.

Ngược lại, lớp tích chập thứ hai bỏ qua đệm,
do đó chiều cao và chiều rộng đều giảm bốn điểm ảnh.
Khi chúng ta đi lên ngăn xếp các lớp,
số kênh tăng lớp-qua-lớp
từ 1 ở đầu vào đến 6 sau lớp tích chập đầu tiên
và 16 sau lớp tích chập thứ hai.
Tuy nhiên, mỗi lớp gộp giảm một nửa chiều cao và chiều rộng.
Cuối cùng, mỗi lớp kết nối đầy đủ giảm chiều,
cuối cùng phát ra đầu ra có chiều
phù hợp với số lớp.


## Huấn luyện

Bây giờ chúng ta đã triển khai mô hình,
hãy [**chạy thử nghiệm để xem mô hình LeNet-5 hoạt động như thế nào trên Fashion-MNIST**].

Trong khi CNN có ít tham số hơn,
chúng vẫn có thể tốn kém hơn để tính toán
so với các MLP sâu tương tự
vì mỗi tham số tham gia vào nhiều phép nhân hơn.
Nếu bạn có quyền truy cập vào GPU, đây có thể là thời điểm tốt
để đưa nó vào sử dụng để tăng tốc huấn luyện.
Lưu ý rằng
lớp `d2l.Trainer` xử lý tất cả các chi tiết.
Theo mặc định, nó khởi tạo các tham số mô hình trên
các thiết bị có sẵn.
Cũng như với MLP, hàm mất mát của chúng ta là entropy chéo,
và chúng ta tối thiểu hóa nó thông qua gradient descent ngẫu nhiên theo minibatch.


## Tóm tắt

Chúng ta đã tiến bộ đáng kể trong chương này. Chúng ta đã chuyển từ các MLP của những năm 1980 sang các CNN của những năm 1990 và đầu 2000. Các kiến trúc được đề xuất, ví dụ: dưới dạng LeNet-5 vẫn có ý nghĩa, cho đến ngày nay. Đáng để so sánh tỷ lệ lỗi trên Fashion-MNIST có thể đạt được với LeNet-5 với những tỷ lệ tốt nhất có thể với MLP ([sec_mlp-implementation](#sec_mlp-implementation)) và những tỷ lệ với các kiến trúc tiên tiến hơn đáng kể như ResNet ([sec_resnet](#sec_resnet)). LeNet tương tự với cái sau hơn là cái trước. Một trong những sự khác biệt chính, như chúng ta sẽ thấy, là lượng tính toán lớn hơn cho phép các kiến trúc phức tạp hơn đáng kể.

Sự khác biệt thứ hai là sự dễ dàng tương đối mà chúng ta có thể triển khai LeNet. Những gì từng là một thách thức kỹ thuật đáng giá nhiều tháng mã C++ và assembly, kỹ thuật để cải thiện SN, một công cụ deep learning dựa trên Lisp sơ khai [Bottou.Le-Cun.1988], và cuối cùng là thử nghiệm với các mô hình hiện có thể được thực hiện trong vài phút. Chính năng suất tăng vọt đáng kinh ngạc này đã dân chủ hóa phát triển mô hình deep learning rất nhiều. Trong chương tiếp theo, chúng ta sẽ đi vào chiếc hố thỏ này để xem nó đưa chúng ta đến đâu.

## Bài tập

1. Hãy hiện đại hóa LeNet. Triển khai và kiểm tra các thay đổi sau:
    1. Thay thế gộp trung bình bằng gộp tối đa.
    1. Thay thế lớp softmax bằng ReLU.
1. Thử thay đổi kích thước của mạng kiểu LeNet để cải thiện độ chính xác của nó ngoài gộp tối đa và ReLU.
    1. Điều chỉnh kích thước cửa sổ tích chập.
    1. Điều chỉnh số kênh đầu ra.
    1. Điều chỉnh số lớp tích chập.
    1. Điều chỉnh số lớp kết nối đầy đủ.
    1. Điều chỉnh tốc độ học và các chi tiết huấn luyện khác (ví dụ: khởi tạo và số epoch).
1. Thử mạng cải tiến trên bộ dữ liệu MNIST gốc.
1. Hiển thị các kích hoạt của lớp đầu tiên và lớp thứ hai của LeNet cho các đầu vào khác nhau (ví dụ: áo len và áo khoác).
1. Điều gì xảy ra với các kích hoạt khi bạn đưa các hình ảnh hoàn toàn khác vào mạng (ví dụ: mèo, ô tô, hoặc thậm chí nhiễu ngẫu nhiên)?


[Discussions](https://discuss.d2l.ai/t/74)
