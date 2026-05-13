# Các Thuật Toán Tối Ưu Hóa
<a id="chap_optimization"></a>

Nếu bạn đọc cuốn sách theo thứ tự đến đây, bạn đã sử dụng một số thuật toán tối ưu hóa để huấn luyện các mô hình deep learning.
Chúng là các công cụ cho phép chúng ta tiếp tục cập nhật các tham số mô hình và tối thiểu hóa giá trị của hàm mất mát, được đánh giá trên tập huấn luyện. Thực vậy, bất kỳ ai hài lòng với việc coi tối ưu hóa là một thiết bị hộp đen để tối thiểu hóa các hàm mục tiêu trong một cài đặt đơn giản có thể tự hài lòng với kiến thức rằng tồn tại một loạt các quy trình như vậy (với các tên như "SGD" và "Adam").

Tuy nhiên, để làm tốt, cần có một số kiến thức sâu hơn.
Các thuật toán tối ưu hóa rất quan trọng cho deep learning.
Một mặt, việc huấn luyện một mô hình deep learning phức tạp có thể mất hàng giờ, hàng ngày, hoặc thậm chí hàng tuần.
Hiệu suất của thuật toán tối ưu hóa ảnh hưởng trực tiếp đến hiệu quả huấn luyện của mô hình.
Mặt khác, hiểu các nguyên tắc của các thuật toán tối ưu hóa khác nhau và vai trò của các siêu tham số của chúng
sẽ cho phép chúng ta điều chỉnh các siêu tham số một cách có mục tiêu để cải thiện hiệu suất của các mô hình deep learning.

Trong chương này, chúng ta khám phá sâu các thuật toán tối ưu hóa deep learning phổ biến.
Hầu hết tất cả các bài toán tối ưu hóa phát sinh trong deep learning đều là *không lồi*.
Tuy nhiên, thiết kế và phân tích các thuật toán trong bối cảnh của các bài toán *lồi* đã được chứng minh là rất hữu ích.
Đó là lý do tại sao chương này bao gồm một giới thiệu về tối ưu hóa lồi và bằng chứng cho một thuật toán gradient giảm ngẫu nhiên rất đơn giản trên hàm mục tiêu lồi.

```toc
:maxdepth: 2

optimization-intro
convexity
gd
sgd
minibatch-sgd
momentum
adagrad
rmsprop
adadelta
adam
lr-scheduler
```
