# Mạng Nơ-ron Tuyến tính cho Hồi quy
<a id="chap_regression"></a>

Trước khi lo lắng về việc làm cho mạng nơ-ron của ta sâu hơn,
sẽ hữu ích khi triển khai một vài mạng nông,
trong đó các đầu vào kết nối trực tiếp với đầu ra.
Điều này sẽ quan trọng vì một vài lý do.
Đầu tiên, thay vì bị phân tâm bởi các kiến trúc phức tạp,
ta có thể tập trung vào những điều cơ bản của huấn luyện mạng nơ-ron,
bao gồm tham số hóa lớp đầu ra, xử lý dữ liệu,
chỉ định hàm mất mát và huấn luyện mô hình.
Thứ hai, lớp mạng nông này vừa hay
bao gồm tập hợp các mô hình tuyến tính,
bao hàm nhiều phương pháp dự đoán thống kê cổ điển,
bao gồm hồi quy tuyến tính và hồi quy softmax.
Hiểu các công cụ cổ điển này rất quan trọng
vì chúng được sử dụng rộng rãi trong nhiều ngữ cảnh
và ta thường cần dùng chúng làm đường cơ sở
khi biện minh cho việc sử dụng các kiến trúc phức tạp hơn.
Chương này sẽ tập trung hẹp vào hồi quy tuyến tính
và chương tiếp theo sẽ mở rộng kho mô hình của ta
bằng cách phát triển mạng nơ-ron tuyến tính cho phân loại.

```toc
:maxdepth: 2

linear-regression
oo-design
synthetic-regression-data
linear-regression-scratch
linear-regression-concise
generalization
weight-decay
```
