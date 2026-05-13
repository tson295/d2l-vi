# Hướng dẫn Xây dựng
<a id="chap_computation"></a>

Cùng với các tập dữ liệu khổng lồ và phần cứng mạnh mẽ,
các công cụ phần mềm tuyệt vời đã đóng vai trò không thể thiếu
trong sự tiến bộ nhanh chóng của deep learning.
Bắt đầu từ thư viện Theano đột phá được phát hành vào năm 2007,
các công cụ mã nguồn mở linh hoạt đã cho phép các nhà nghiên cứu
nhanh chóng tạo nguyên mẫu các mô hình, tránh công việc lặp đi lặp lại
khi tái sử dụng các thành phần tiêu chuẩn
trong khi vẫn duy trì khả năng thực hiện các sửa đổi cấp thấp.
Theo thời gian, các thư viện deep learning đã phát triển
để cung cấp các trừu tượng hóa ngày càng thô hơn.
Giống như các nhà thiết kế bán dẫn đã đi từ việc chỉ định transistor
đến các mạch logic cho đến viết code,
các nhà nghiên cứu mạng nơ-ron đã chuyển từ việc nghĩ về
hành vi của các nơ-ron nhân tạo riêng lẻ
đến việc quan niệm mạng theo các lớp toàn bộ,
và bây giờ thường thiết kế kiến trúc với các *khối* thô hơn nhiều trong đầu.


Cho đến nay, chúng ta đã giới thiệu một số khái niệm machine learning cơ bản,
leo dần đến các mô hình deep learning hoạt động đầy đủ.
Trong chương trước,
chúng ta đã cài đặt từng thành phần của một MLP từ đầu
và thậm chí cho thấy cách tận dụng các API cấp cao
để triển khai cùng các mô hình đó một cách dễ dàng.
Để đưa bạn đến đó nhanh chóng, chúng ta đã *gọi đến* các thư viện,
nhưng bỏ qua các chi tiết nâng cao hơn về *cách chúng hoạt động*.
Trong chương này, chúng ta sẽ vén màn bí mật,
đào sâu hơn vào các thành phần chính của tính toán deep learning,
cụ thể là xây dựng mô hình, truy cập và khởi tạo tham số,
thiết kế các lớp và khối tùy chỉnh, đọc và ghi mô hình vào đĩa,
và tận dụng GPU để đạt được tốc độ tăng đáng kể.
Những hiểu biết này sẽ đưa bạn từ *người dùng cuối* đến *người dùng nâng cao*,
cung cấp cho bạn các công cụ cần thiết để hưởng lợi từ
một thư viện deep learning trưởng thành trong khi vẫn giữ được tính linh hoạt
để cài đặt các mô hình phức tạp hơn, bao gồm cả những mô hình bạn tự phát minh!
Mặc dù chương này không giới thiệu bất kỳ mô hình hay tập dữ liệu mới nào,
các chương mô hình hóa nâng cao tiếp theo phụ thuộc nhiều vào các kỹ thuật này.

```toc
:maxdepth: 2

model-construction
parameters
init-param
lazy-init
custom-layer
read-write
use-gpu
```
