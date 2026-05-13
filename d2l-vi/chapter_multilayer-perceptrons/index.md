# Perceptron Đa lớp
<a id="chap_perceptrons"></a>

Trong chương này, chúng ta sẽ giới thiệu mạng *sâu* thực sự đầu tiên của bạn.
Các mạng sâu đơn giản nhất được gọi là *perceptron đa lớp*,
và chúng bao gồm nhiều lớp nơ-ron
mỗi lớp được kết nối đầy đủ với các nơ-ron trong lớp dưới
(từ đó chúng nhận đầu vào)
và lớp trên (mà chúng, lần lượt, ảnh hưởng đến).
Mặc dù vi phân tự động
đơn giản hóa đáng kể việc cài đặt các thuật toán deep learning,
chúng ta sẽ đi sâu vào cách tính các gradient này
trong các mạng sâu.
Sau đó, chúng ta sẽ
sẵn sàng
thảo luận về các vấn đề liên quan đến ổn định số và khởi tạo tham số
là chìa khóa để huấn luyện thành công các mạng sâu.
Khi chúng ta huấn luyện các mô hình có dung lượng cao như vậy, chúng ta có nguy cơ quá khớp. Do đó, chúng ta sẽ
xem xét lại chuẩn hóa và tổng quát hóa
cho các mạng sâu.
Xuyên suốt, chúng ta nhằm
cung cấp cho bạn sự hiểu biết vững chắc không chỉ về các khái niệm mà còn về thực hành sử dụng các mạng sâu.
Ở cuối chương này, chúng ta áp dụng những gì chúng ta đã giới thiệu cho đến nay vào một trường hợp thực tế: dự đoán giá nhà. Chúng ta để các vấn đề liên quan đến hiệu suất tính toán, khả năng mở rộng, và hiệu quả
của mô hình cho các chương tiếp theo.

```toc
:maxdepth: 2

mlp
mlp-implementation
backprop
numerical-stability-and-init
generalization-deep
dropout
kaggle-house-price
```
