# Mạng Sử dụng Khối (VGG)
<a id="sec_vgg"></a>

Trong khi AlexNet cung cấp bằng chứng thực nghiệm rằng CNN sâu
có thể đạt được kết quả tốt, nó không cung cấp một mẫu tổng quát
để hướng dẫn các nhà nghiên cứu tiếp theo trong việc thiết kế các mạng mới.
Trong các phần tiếp theo, chúng ta sẽ giới thiệu một số khái niệm heuristic
thường được sử dụng để thiết kế các mạng sâu.

Tiến bộ trong lĩnh vực này phản chiếu tiến bộ của VLSI (tích hợp quy mô rất lớn)
trong thiết kế chip
nơi các kỹ sư chuyển từ đặt bóng bán dẫn
sang các phần tử logic đến các khối logic [Mead.1980].
Tương tự, thiết kế kiến trúc mạng nơ-ron
đã trở nên trừu tượng hơn theo tiến trình,
với các nhà nghiên cứu chuyển từ suy nghĩ về
các nơ-ron riêng lẻ sang toàn bộ các lớp,
và bây giờ là các khối, các mẫu lặp lại của các lớp. Một thập kỷ sau, điều này đã
tiến đến việc các nhà nghiên cứu sử dụng toàn bộ các mô hình đã được huấn luyện để tái sử dụng chúng cho các tác vụ khác nhau,
mặc dù có liên quan. Các mô hình lớn được huấn luyện sẵn như vậy thường được gọi là
*mô hình nền tảng* [bommasani2021opportunities].

Quay lại thiết kế mạng. Ý tưởng sử dụng các khối lần đầu tiên xuất hiện từ
Visual Geometry Group (VGG) tại Đại học Oxford,
trong mạng *VGG* mang tên họ [Simonyan.Zisserman.2014].
Dễ dàng triển khai các cấu trúc lặp lại này trong code
bằng bất kỳ framework deep learning hiện đại nào bằng cách sử dụng vòng lặp và chương trình con.


```python
from d2l import torch as d2l
import torch
from torch import nn
```


## (**Khối VGG**)
<a id="subsec_vgg-blocks"></a>

Khối xây dựng cơ bản của CNN
là một chuỗi gồm các thành phần sau:
(i) một lớp tích chập
với đệm để duy trì độ phân giải,
(ii) một phi tuyến tính như ReLU,
(iii) một lớp gộp như
gộp tối đa để giảm độ phân giải. Một trong những vấn đề với
cách tiếp cận này là độ phân giải không gian giảm khá nhanh. Đặc biệt,
điều này đặt ra giới hạn cứng $\log_2 d$ lớp tích chập trên mạng trước khi tất cả
các chiều ($d$) được sử dụng hết. Ví dụ, trong trường hợp ImageNet, không thể có
nhiều hơn 8 lớp tích chập theo cách này.

Ý tưởng chính của Simonyan.Zisserman.2014 là sử dụng *nhiều* phép tích chập giữa các lần lấy mẫu thấp hơn
qua gộp tối đa dưới dạng khối. Họ chủ yếu quan tâm đến việc mạng sâu hay
rộng hoạt động tốt hơn. Ví dụ, việc áp dụng liên tiếp hai phép tích chập $3 \times 3$
chạm đến cùng các điểm ảnh như một phép tích chập $5 \times 5$ duy nhất. Đồng thời, phép sau sử dụng xấp xỉ
nhiều tham số ($25 \cdot c^2$) như ba phép tích chập $3 \times 3$ ($3 \cdot 9 \cdot c^2$).
Trong phân tích khá chi tiết, họ đã chỉ ra rằng các mạng sâu và hẹp vượt qua đáng kể so với các mạng nông tương đương. Điều này đặt deep learning vào một cuộc tìm kiếm các mạng ngày càng sâu hơn với hơn 100 lớp cho các ứng dụng điển hình.
Xếp chồng các phép tích chập $3 \times 3$
đã trở thành tiêu chuẩn vàng trong các mạng sâu sau này (một quyết định thiết kế chỉ được xem xét lại gần đây bởi
liu2022convnet). Do đó, các triển khai nhanh cho các phép tích chập nhỏ đã trở thành tiêu chuẩn trên GPU [lavin2016fast].

Quay lại VGG: một khối VGG bao gồm một *chuỗi* các phép tích chập với nhân $3\times3$ và đệm là 1
(giữ nguyên chiều cao và chiều rộng) theo sau là lớp gộp tối đa $2 \times 2$ với sải bước là 2
(giảm một nửa chiều cao và chiều rộng sau mỗi khối).
Trong code dưới đây, chúng ta định nghĩa một hàm gọi là `vgg_block`
để triển khai một khối VGG.

Hàm dưới đây nhận hai đối số,
tương ứng với số lớp tích chập `num_convs`
và số kênh đầu ra `num_channels`.


```python
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```


## [**Mạng VGG**]
<a id="subsec_vgg-network"></a>

Giống như AlexNet và LeNet,
Mạng VGG có thể được chia thành hai phần:
phần đầu bao gồm chủ yếu là các lớp tích chập và gộp
và phần thứ hai bao gồm các lớp kết nối đầy đủ giống với những lớp trong AlexNet.
Sự khác biệt chính là
các lớp tích chập được nhóm trong các phép biến đổi phi tuyến tính
để lại chiều không thay đổi, theo sau là bước giảm độ phân giải, như
được mô tả trong [fig_vgg](#fig_vgg).

![Từ AlexNet sang VGG. Sự khác biệt chính là VGG bao gồm các khối lớp, trong khi các lớp của AlexNet đều được thiết kế riêng lẻ.](../img/vgg.svg)
<a id="fig_vgg"></a>

Phần tích chập của mạng kết nối một số khối VGG từ [fig_vgg](#fig_vgg) (cũng được định nghĩa trong hàm `vgg_block`)
lần lượt. Việc nhóm các phép tích chập này là một mẫu đã
gần như không thay đổi trong thập kỷ qua, mặc dù lựa chọn cụ thể của
các phép toán đã trải qua những sửa đổi đáng kể.
Biến `arch` bao gồm một danh sách các tuple (một cho mỗi khối),
trong đó mỗi cái chứa hai giá trị: số lớp tích chập
và số kênh đầu ra,
đây chính xác là các đối số cần để gọi
hàm `vgg_block`. Do đó, VGG định nghĩa một *họ* mạng thay vì chỉ
một biểu hiện cụ thể. Để xây dựng một mạng cụ thể, chúng ta chỉ đơn giản lặp qua `arch` để tạo thành các khối.


Mạng VGG gốc có năm khối tích chập,
trong đó hai khối đầu tiên mỗi khối có một lớp tích chập
và ba khối còn lại mỗi khối có hai lớp tích chập.
Khối đầu tiên có 64 kênh đầu ra
và mỗi khối tiếp theo tăng gấp đôi số kênh đầu ra,
cho đến khi con số đó đạt 512.
Vì mạng này sử dụng tám lớp tích chập
và ba lớp kết nối đầy đủ, nó thường được gọi là VGG-11.


Như bạn có thể thấy, chúng ta giảm một nửa chiều cao và chiều rộng tại mỗi khối,
cuối cùng đạt chiều cao và chiều rộng là 7
trước khi làm phẳng các biểu diễn
để xử lý bởi phần kết nối đầy đủ của mạng.
Simonyan.Zisserman.2014 đã mô tả một số biến thể khác của VGG.
Trên thực tế, việc đề xuất *họ* mạng với
sự đánh đổi tốc độ-độ chính xác khác nhau khi giới thiệu một kiến trúc mới đã trở thành tiêu chuẩn.

## Huấn luyện

[**Vì VGG-11 đòi hỏi tính toán nhiều hơn AlexNet
chúng ta xây dựng một mạng với số lượng kênh nhỏ hơn.**]
Điều này là quá đủ để huấn luyện trên Fashion-MNIST.
Quá trình [**huấn luyện mô hình**] tương tự như AlexNet trong [sec_alexnet](#sec_alexnet).
Một lần nữa quan sát sự khớp chặt chẽ giữa mất mát kiểm định và huấn luyện,
gợi ý chỉ có một lượng nhỏ quá khớp.


## Tóm tắt

Người ta có thể lập luận rằng VGG là mạng nơ-ron tích chập hiện đại thực sự đầu tiên. Trong khi AlexNet giới thiệu nhiều thành phần làm cho deep learning hiệu quả ở quy mô, chính VGG đã giới thiệu các tính chất quan trọng như các khối của nhiều phép tích chập và ưu tiên cho các mạng sâu và hẹp. Đây cũng là mạng đầu tiên thực sự là cả một họ các mô hình được tham số hóa tương tự, mang lại cho nhà thực hành sự đánh đổi phong phú giữa độ phức tạp và tốc độ. Đây cũng là nơi các framework deep learning hiện đại tỏa sáng. Không còn cần thiết phải tạo các tệp cấu hình XML để chỉ định một mạng mà thay vào đó là lắp ráp các mạng đó thông qua code Python đơn giản.

Gần đây hơn, ParNet [Goyal.Bochkovskiy.Deng.ea.2021] đã chứng minh rằng có thể đạt được hiệu suất cạnh tranh bằng cách sử dụng kiến trúc nông hơn nhiều thông qua số lượng lớn các phép tính song song. Đây là sự phát triển thú vị và có hy vọng rằng nó sẽ ảnh hưởng đến các thiết kế kiến trúc trong tương lai. Tuy nhiên, trong phần còn lại của chương, chúng ta sẽ theo con đường tiến bộ khoa học trong thập kỷ qua.

## Bài tập

1. So với AlexNet, VGG chậm hơn nhiều về mặt tính toán và cũng cần nhiều bộ nhớ GPU hơn.
    1. So sánh số lượng tham số cần thiết cho AlexNet và VGG.
    1. So sánh số lượng phép toán dấu phẩy động được sử dụng trong các lớp tích chập và trong các lớp kết nối đầy đủ.
    1. Làm thế nào bạn có thể giảm chi phí tính toán được tạo ra bởi các lớp kết nối đầy đủ?
1. Khi hiển thị các chiều liên quan đến các lớp khác nhau của mạng, chúng ta chỉ thấy thông tin liên quan đến tám khối (cộng với một số phép biến đổi phụ), mặc dù mạng có 11 lớp. Ba lớp còn lại đã đi đâu?
1. Sử dụng Bảng 1 trong bài báo VGG [Simonyan.Zisserman.2014] để xây dựng các mô hình phổ biến khác, chẳng hạn như VGG-16 hoặc VGG-19.
1. Nâng mẫu độ phân giải trong Fashion-MNIST gấp tám lần từ $28 \times 28$ lên kích thước $224 \times 224$ rất lãng phí. Thử sửa đổi kiến trúc mạng và chuyển đổi độ phân giải, ví dụ: sang kích thước 56 hoặc 84 cho đầu vào của nó. Bạn có thể làm điều đó mà không giảm độ chính xác của mạng không? Tham khảo bài báo VGG [Simonyan.Zisserman.2014] để có ý tưởng về việc thêm nhiều phi tuyến tính hơn trước khi lấy mẫu thấp hơn.


[Discussions](https://discuss.d2l.ai/t/78)
